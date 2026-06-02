import sys, importlib, traceback
sys.path.insert(0, '.')

# Re-load from a clean state
for k in list(sys.modules):
    if k.startswith(('api', 'src')):
        del sys.modules[k]

modules = [
    'src',
    'src.config',
    'src.logger',
    'src.progress',
    'src.data',
    'src.data.loader',
    'src.data.splitter',
    'src.data.encoders',
    'src.evaluation',
    'src.evaluation.metrics',
    'src.evaluation.evaluator',
    'src.models',
    'src.models.fpgrowth_model',
    'src.models.als_model',
    'src.visualization',
    'src.visualization._style',
    'src.visualization.training_viz',
    'src.visualization.inference_viz',
    'src.pipeline',
    'src.pipeline.train_pipeline',
    'src.pipeline.infer_pipeline',
    'api',
    'api.schemas',
    'api.routes',
    'api.routes.health',
    'api.routes.train',
    'api.routes.recommend',
    'api.routes.metrics',
    'api.main',
]

fails = 0
for m in modules:
    try:
        importlib.import_module(m)
        print(f'  ok: {m}')
    except Exception as e:
        fails += 1
        print(f'  FAIL: {m}: {e}')
        traceback.print_exc()
print('---')
print('FAILED' if fails else 'ALL OK')

