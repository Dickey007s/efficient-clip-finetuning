import open_clip, torch

model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', quick_gelu=True)
visual = model.visual

print('Visual type:', type(visual))
print('Visual attributes:', [a for a in dir(visual) if not a.startswith('_')])
print('Transformer type:', type(visual.transformer))
print('Resblocks:', len(visual.transformer.resblocks))

block = visual.transformer.resblocks[0]
print('Block type:', type(block))
print('Block attributes:', [a for a in dir(block) if not a.startswith('_')])

print('Has attn:', hasattr(block, 'attn'))
if hasattr(block, 'attn'):
    attn = block.attn
    print('Attn type:', type(attn))
    print('Attn attributes:', [a for a in dir(attn) if not a.startswith('_')])
    if hasattr(attn, 'in_proj_weight'):
        print('in_proj_weight shape:', attn.in_proj_weight.shape)
    if hasattr(attn, 'out_proj'):
        print('out_proj type:', type(attn.out_proj))
        print('out_proj weight shape:', attn.out_proj.weight.shape)

print('Has mlp:', hasattr(block, 'mlp'))
if hasattr(block, 'mlp'):
    mlp = block.mlp
    print('MLP type:', type(mlp))
    print('MLP attributes:', [a for a in dir(mlp) if not a.startswith('_')])
    if hasattr(mlp, 'c_fc'):
        print('c_fc weight shape:', mlp.c_fc.weight.shape)
    if hasattr(mlp, 'c_proj'):
        print('c_proj weight shape:', mlp.c_proj.weight.shape)
