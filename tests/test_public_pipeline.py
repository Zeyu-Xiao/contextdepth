"""Public RGB behavior is independent of paper manifests, ROI crops and scoring."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from contextdepth import ContextDepthPipeline
from contextdepth.camera import crop_prediction
from contextdepth.cli import parser
from contextdepth.config import load_runtime_config
from contextdepth.generation import OutpaintingBaseline
from contextdepth.reference import Reference
from contextdepth.types import CameraEstimate, Prediction


class FakeGenerator(OutpaintingBaseline):
    def generate_images(self, rgb, *, bbox, canvas_size, seeds):
        width, height = canvas_size
        return [np.full((height, width, 3), seed + 11, np.uint8) for seed in seeds]


class FakeDepth:
    supports_setting_reuse = True
    instances = []

    def __init__(self, name, paths, device):
        self.name, self.paths, self.setting = name, dict(paths), paths['setting']
        self.max_batch_size = paths['batch_size']
        self.calls, self.settings = [], [self.setting]
        self.closed = False
        type(self).instances.append(self)

    def configure(self, name, paths):
        self.name, self.paths, self.setting = name, dict(paths), paths['setting']
        self.settings.append(self.setting)

    def predict(self, requests):
        self.calls.append(requests)
        predictions = []
        for request in requests:
            height, width = request.rgb.shape[:2]
            k = request.camera if request.camera is not None else np.array([[20,0,width/2],[0,20,height/2],[0,0,1]])
            predictions.append(Prediction(np.full((height,width),2,np.float32), CameraEstimate(intrinsics=k.copy())))
        return predictions

    def close(self):
        self.closed = True


class PublicPipelineTests(unittest.TestCase):
    def setUp(self):
        FakeDepth.instances.clear()
        self.rgb = np.arange(10*20*3,dtype=np.uint8).reshape(10,20,3)
        self.k = np.array([[30.,0.,10.],[0.,31.,5.],[0.,0.,1.]])

    def pipeline(self, **kwargs):
        return ContextDepthPipeline(depth_model=FakeDepth, outpainter=FakeGenerator, device='cpu', **kwargs)

    def test_default_full_outputs_and_no_reference(self):
        with self.pipeline() as pipeline:
            result = pipeline(self.rgb)
            self.assertIsNone(result.selected)
            self.assertIsNone(result.reference_uid)
            self.assertEqual(result.direct.depth.shape,(10,20))
            self.assertEqual([c.outpainting.seed for c in result.candidates],[0,1,2])
            for candidate in result.candidates:
                self.assertEqual(candidate.prediction.depth.shape,(12,24))
                self.assertEqual(candidate.outpainting.bbox,(2,1,22,11))
                np.testing.assert_array_equal(candidate.outpainting.rgb[1:11,2:22],self.rgb)
            for request in FakeDepth.instances[0].calls[0]:
                self.assertEqual(request.geometry.roi_x,0)
                self.assertEqual(request.geometry.roi_width,request.rgb.shape[1])
                self.assertFalse(hasattr(request,'depth'))

    def test_camera_is_mapped_once_to_each_complete_rgb(self):
        with self.pipeline() as pipeline:
            pipeline(self.rgb,camera=self.k)
            requests=FakeDepth.instances[0].calls[0]
            np.testing.assert_array_equal(requests[0].camera,self.k)
            expected=self.k.copy();expected[0,2]+=2;expected[1,2]+=1
            for request in requests[1:]:
                np.testing.assert_array_equal(request.camera,expected)
            np.testing.assert_array_equal(self.k,[[30,0,10],[0,31,5],[0,0,1]])

    def test_generation_only_does_not_load_depth(self):
        with self.pipeline() as pipeline:
            pipeline.generate(self.rgb)
            self.assertEqual(FakeDepth.instances,[])

    def test_depth_only_needs_no_generator_or_reference(self):
        with patch('contextdepth.pipeline.create_outpainter',side_effect=AssertionError('generator loaded')):
            with self.pipeline() as pipeline:
                pred=pipeline.estimate_depth(self.rgb)
                self.assertEqual(pred.depth.shape,(10,20))

    def test_same_depth_instance_serves_known_and_unknown_inputs(self):
        with self.pipeline() as pipeline:
            pipeline.estimate_depth(self.rgb)
            pipeline.estimate_depth(self.rgb,camera=self.k)
            pipeline.estimate_depth(self.rgb)
            self.assertEqual(len(FakeDepth.instances),1)
            self.assertEqual(FakeDepth.instances[0].settings,['unknown','known','unknown'])
        self.assertTrue(FakeDepth.instances[0].closed)

    def test_preloaded_instance_keeps_its_batch_size_and_caller_ownership(self):
        engine = FakeDepth('custom', {'setting':'unknown', 'batch_size':4}, 'cpu')
        with ContextDepthPipeline(depth_model=engine, device='cpu') as pipeline:
            pipeline.estimate_depth(self.rgb, camera=self.k)
            self.assertEqual(engine.paths['batch_size'],4)
        self.assertFalse(engine.closed)
        with self.assertRaisesRegex(ValueError,'batch size'):
            ContextDepthPipeline(depth_model=engine, depth_batch_size=1, device='cpu')

    def test_optional_bank_calibrates_reference_once_without_query_gt(self):
        bank = Mock()
        bank.retrieve.return_value = Reference('diode-reference', self.rgb.copy(),
                                               np.ones((10,20),np.float32),self.k.copy())
        with patch('contextdepth.reference.DIODEReferenceBank',return_value=bank):
            with self.pipeline(reference_bank='diode',diode_root='downloaded') as pipeline:
                unknown = pipeline(self.rgb)
                known = pipeline(self.rgb,camera=self.k)
                self.assertEqual(unknown.reference_uid,'diode-reference')
                self.assertEqual(unknown.selected.depth.shape,(10,20))
                self.assertEqual(unknown.selection.selected_arm,'candidate0')
                self.assertTrue(known.selection.is_direct)
                self.assertEqual(len(FakeDepth.instances),1)
                calls = FakeDepth.instances[0].calls
                self.assertEqual([len(batch) for batch in calls],[4,1,4])
                self.assertTrue(all(r.camera is None for r in calls[0]))
                np.testing.assert_array_equal(calls[1][0].camera,self.k)
                self.assertEqual(bank.retrieve.call_count,2)
        bank.close.assert_called_once()

    def test_saved_context_bypasses_generation(self):
        contexts=FakeGenerator().generate(self.rgb,expansion=(.2,.3,.4,.5))
        with self.pipeline() as pipeline, patch.object(pipeline,'generate',side_effect=AssertionError('regenerated')):
            result=pipeline(self.rgb,outpaintings=contexts)
            self.assertEqual(result.candidates[0].prediction.depth.shape,(15,38))

    def test_context_cannot_change_query(self):
        contexts=FakeGenerator().generate(self.rgb)
        contexts[0].rgb[1,2]=0
        with self.pipeline() as pipeline:
            with self.assertRaisesRegex(ValueError,'preserve'):
                pipeline(self.rgb,outpaintings=contexts)

    def test_outputs_save_complete_candidates_and_no_selected_file(self):
        with self.pipeline() as pipeline,tempfile.TemporaryDirectory() as directory:
            result=pipeline(self.rgb);root=Path(directory)
            (root/'selected.npz').write_bytes(b'stale prior selection')
            result.save(root)
            self.assertFalse((root/'selected.npz').exists())
            with np.load(root/'candidate0.npz') as data:
                self.assertEqual(data['depth'].shape,(12,24))
                self.assertEqual(data['intrinsics'].shape,(3,3))
            self.assertIsNone(json.loads((root/'result.json').read_text())['selected_arm'])

    def test_non_diode_banks_rejected_and_diode_requires_download(self):
        with self.assertRaisesRegex(ValueError,'Only'):
            self.pipeline(reference_bank='custom')
        with self.assertRaisesRegex(ValueError,'diode_root'):
            self.pipeline(reference_bank='diode')
        with self.assertRaises((ValueError,FileNotFoundError)):
            self.pipeline(reference_bank='diode',diode_root='/missing/contextdepth/diode')

    def test_crop_generated_border_maps_camera_and_rays_without_resize(self):
        rays=np.ones((3,12,24),np.float32)
        k=self.k.copy();k[0,2]+=2;k[1,2]+=1
        pred=Prediction(np.full((12,24),2,np.float32),CameraEstimate(intrinsics=k,rays=rays))
        cropped=crop_prediction(pred,(2,1,22,11))
        self.assertEqual(cropped.depth.shape,(10,20))
        self.assertEqual(cropped.camera.rays.shape,(3,10,20))
        np.testing.assert_array_equal(cropped.camera.intrinsics,self.k)
        self.assertEqual(pred.camera.intrinsics[0,2],12)

    def test_import_and_runtime_config_do_not_import_evaluation_or_torch(self):
        code="from contextdepth import ContextDepthPipeline; import sys; assert 'torch' not in sys.modules; assert not any(x == 'evaluation' or x.startswith('evaluation.') for x in sys.modules)"
        subprocess.run([sys.executable,'-B','-c',code],check=True)
        self.assertNotIn('experiments',load_runtime_config())

    def test_normal_cli_has_no_evaluation_manifest_requirement(self):
        args=parser().parse_args(['infer','--image','q.png','--output','output'])
        self.assertFalse(hasattr(args,'manifest'))
        self.assertIsNone(args.reference_bank)
        self.assertIsNone(args.depth_model)

    def test_bad_camera_fails_before_model_loading(self):
        with self.pipeline() as pipeline:
            with self.assertRaises(ValueError):pipeline.estimate_depth(self.rgb,camera=np.eye(4))
            self.assertEqual(FakeDepth.instances,[])

    def test_custom_file_adapter_does_not_need_yaml_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'new_model.py'
            # The isolated file adapter is exercised using a self-contained
            # implementation so it does not rely on tests being a package.
            source.write_text('import numpy as np\nfrom contextdepth.types import Prediction\nclass Baseline:\n def __init__(self,name,paths,device): self.setting=paths["setting"]\n def predict(self,requests): return [Prediction(np.ones(r.rgb.shape[:2],np.float32)) for r in requests]\n')
            with ContextDepthPipeline(depth_model=str(source)+':Baseline',device='cpu') as pipeline:
                self.assertEqual(pipeline.estimate_depth(self.rgb).depth.shape,(10,20))


if __name__=='__main__':
    unittest.main()
