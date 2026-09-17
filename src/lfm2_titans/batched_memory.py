"""Per-row memory ports around one genuinely batched native VL forward."""
import torch


class BatchedMemoryBlock:
    def __init__(self,model,units,input_ids,spatial_shapes=None,observation_mask=None,fresh_rows=None):
        self.model=model
        self.blocks=[model.memory.begin(unit.graph) for unit in units]
        self.observation_mask=observation_mask
        self.fresh_rows=fresh_rows or [False]*len(units)
        # Native vision packs tiles from all images into its leading dimension.
        # Attribute each complete tile to the row containing its image tokens.
        self.tiles=[[] for _ in units]
        if spatial_shapes is not None:
            factor=model.config.downsample_factor
            counts=((spatial_shapes[:,0]//factor)*(spatial_shapes[:,1]//factor)).tolist()
            wanted=(input_ids==model.config.image_token_id).sum(-1).tolist()
            cursor=0
            for row,count in enumerate(wanted):
                while count:
                    if cursor>=len(counts) or counts[cursor]>count:
                        raise ValueError('image tile boundary does not align with sample')
                    self.tiles[row].append(cursor);count-=counts[cursor];cursor+=1
            if cursor!=len(counts):raise ValueError('unassigned image tiles')

    def _parts(self,name,features,mask=None):
        if name==self.model.config.vision_port:
            for i,indices in enumerate(self.tiles):
                if indices:
                    ids=torch.tensor(indices,device=features.device)
                    yield i,ids,features.index_select(0,ids),None if mask is None else mask.index_select(0,ids)
        else:
            if len(features)!=len(self.blocks):raise ValueError('language memory batch mismatch')
            for i in range(len(self.blocks)):
                valid=None if mask is None else mask[i:i+1]
                if self.observation_mask is not None:
                    valid=self.observation_mask[i:i+1] if valid is None else valid.bool() & self.observation_mask[i:i+1].bool()
                yield i,slice(i,i+1),features[i:i+1],valid

    def observe(self,event,name,features,mask=None):
        for i,_,part,valid in self._parts(name,features,mask):
            if valid is not None and not valid.any():continue
            if valid is not None and int(valid.sum())<2 and name!=self.model.config.vision_port:
                self.blocks[i].native_observations[name]=part[valid.bool()]
                continue
            self.blocks[i].observe(event,name,part,valid)

    def residual(self,name,features):
        if all(self.fresh_rows):
            if self.model.memory.residual_recall:return features
            # Every fresh row reads the identical learned prior. Flattening
            # them through one read preserves row independence and gradients.
            value=self.model.memory.read(name,features,self.model.memory.slow_weights)
            gate=self.model.memory.ports[name].residual_gate.sigmoid()
            return features+(value*gate).to(features.dtype)
        output=features.clone()
        for i,indices,part,_ in self._parts(name,features):
            # A freshly initialized fast graph is exactly the slow prior.
            # Its residual and its parameter derivatives cancel identically.
            if self.fresh_rows[i] and self.model.memory.residual_recall:continue
            value=self.blocks[i].residual(name,part)
            if isinstance(indices,slice):output[indices]=value
            else:output=output.index_copy(0,indices,value)
        return output
