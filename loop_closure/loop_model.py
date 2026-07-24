import torch
from pathlib import Path

from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from pipeline.config import DetectionConfig
from pipeline.manifest import ImageManifest

from .methods.base import LoopCandidate
from .vpr_model import VPRModel


class LoopDetector:
    """Loop detector class for detecting loop closures in image sequences"""

    def __init__(
        self,
        detection_config: DetectionConfig,
        image_manifest: ImageManifest,
        output_path: Path,
    ):
        self.config = detection_config
        self.image_manifest = image_manifest
        self.ckpt_path = detection_config.salad_checkpoint
        self.image_size = detection_config.image_size
        self.batch_size = detection_config.batch_size
        self.similarity_threshold = detection_config.similarity_threshold
        self.top_k = detection_config.top_k
        self.use_nms = detection_config.nms_enabled
        self.nms_threshold = detection_config.nms_frame_radius
        self.output = Path(output_path)
        self._vpr_config = {
            "Weights": {"DINO": detection_config.dino_checkpoint}
        }

        self.model = None
        self.device = None
        self.image_paths = image_manifest.as_strings()
        self.descriptors = None
        self.loop_closures = None

    def _input_transform(self, image_size=None):
        """Create image transformation function"""
        MEAN = [0.485, 0.456, 0.406];
        STD = [0.229, 0.224, 0.225]
        if image_size:
            return T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    def load_model(self):
        """Load model"""
        model = VPRModel(
            backbone_arch='dinov2_vitb14',
            backbone_config={
                'num_trainable_blocks': 4,
                'return_token': True,
                'norm_layer': True,
            },
            agg_arch='SALAD',
            agg_config={
                'num_channels': 768,
                'num_clusters': 64,
                'cluster_dim': 128,
                'token_dim': 256,
            },
            vggt_long_config=self._vpr_config
        )

        model.load_state_dict(
            torch.load(
                self.ckpt_path,
                map_location="cpu",
                weights_only=False,
            )
        )
        model = model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Model loaded: {self.ckpt_path}")

        self.model = model
        self.device = device
        return model, device

    def get_image_paths(self):
        """Return the canonical immutable-manifest snapshot."""
        return self.image_paths

    def extract_descriptors(self):
        """Extract image feature descriptors"""
        if self.model is None or self.device is None:
            self.load_model()

        if self.image_paths is None:
            self.get_image_paths()

        transform = self._input_transform(self.image_size)
        descriptors = []

        for i in tqdm(range(0, len(self.image_paths), self.batch_size), desc="Extracting features"):
            batch_paths = self.image_paths[i:i + self.batch_size]
            batch_imgs = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = transform(img)
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    img = torch.zeros(3, 224, 224) if self.image_size is None else torch.zeros(3, self.image_size[0],
                                                                                               self.image_size[1])
                    batch_imgs.append(img)

            batch_tensor = torch.stack(batch_imgs).to(self.device)

            with torch.no_grad():
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                    batch_descriptors = self.model(batch_tensor).cpu()

            descriptors.append(batch_descriptors)

        self.descriptors = torch.cat(descriptors)
        return self.descriptors

    def _apply_nms_filter(self, loop_closures, nms_threshold):
        """Apply Non-Maximum Suppression (NMS) filtering to loop pairs"""
        if not loop_closures or nms_threshold <= 0:
            return loop_closures

        sorted_loops = sorted(loop_closures, key=lambda x: x[2], reverse=True)
        filtered_loops = []
        suppressed = set()

        max_frame = max(max(idx1, idx2) for idx1, idx2, _ in loop_closures)

        for idx1, idx2, sim in sorted_loops:
            if idx1 in suppressed or idx2 in suppressed:
                continue

            filtered_loops.append((idx1, idx2, sim))

            suppress_range = set()

            start1 = max(0, idx1 - nms_threshold)
            end1 = min(idx1 + nms_threshold + 1, idx2)
            suppress_range.update(range(start1, end1))

            start2 = max(idx1 + 1, idx2 - nms_threshold)
            end2 = min(idx2 + nms_threshold + 1, max_frame + 1)
            suppress_range.update(range(start2, end2))

            suppressed.update(suppress_range)

        return filtered_loops

    def _ensure_decending_order(self, tuples_list):
        return [(max(a, b), min(a, b), score) for a, b, score in tuples_list]

    def find_loop_closures(self):
        """Find loop closures"""
        if self.descriptors is None:
            self.extract_descriptors()

        import faiss

        embed_size = self.descriptors.shape[1]
        faiss_index = faiss.IndexFlatIP(embed_size)

        normalized_descriptors = self.descriptors.numpy()
        faiss_index.add(normalized_descriptors)

        similarities, indices = faiss_index.search(normalized_descriptors,
                                                   self.top_k + 1)  # +1 because self is most similar

        loop_closures = []
        for i in range(len(self.descriptors)):
            # Skip first result (self)
            for j in range(1, self.top_k + 1):
                neighbor_idx = indices[i, j]
                similarity = similarities[i, j]

                if similarity > self.similarity_threshold and abs(i - neighbor_idx) > 10:
                    if i < neighbor_idx:
                        loop_closures.append((i, neighbor_idx, similarity))
                    else:
                        loop_closures.append((neighbor_idx, i, similarity))

        loop_closures = list(set(loop_closures))
        loop_closures.sort(key=lambda x: x[2], reverse=True)

        if self.use_nms and self.nms_threshold > 0:
            loop_closures = self._apply_nms_filter(loop_closures, self.nms_threshold)

        canonical = self._ensure_decending_order(loop_closures)
        self.loop_closures = tuple(
            LoopCandidate(
                frame_a=int(frame_a),
                frame_b=int(frame_b),
                similarity=float(similarity),
            )
            for frame_a, frame_b, similarity in canonical
        )
        return self.loop_closures

    def save_results(self):
        """Save loop detection results to file"""
        if self.loop_closures is None:
            self.find_loop_closures()

        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as f:
            f.write("# Loop Detection Results (index1, index2, similarity)\n")
            if self.use_nms:
                f.write(f"# NMS filtering applied, threshold: {self.nms_threshold}\n")
            f.write("\n# Loop pairs:\n")
            for candidate in self.loop_closures:
                f.write(
                    f"{candidate.frame_a}, {candidate.frame_b}, "
                    f"{candidate.similarity:.4f}\n"
                )
            f.write("\n# Image path list:\n")
            for i, path in enumerate(self.image_paths):
                f.write(f"# {i}: {path}\n")

        print(f"Found {len(self.loop_closures)} loop pairs, results saved to {self.output}")
        if self.use_nms:
            print(f"NMS filtering applied, threshold: {self.nms_threshold}")

        if self.loop_closures:
            print("\nTop 10 loop pairs:")
            for i, candidate in enumerate(self.loop_closures[:10]):
                print(
                    f"{candidate.frame_a}, {candidate.frame_b}, "
                    f"similarity: {candidate.similarity:.4f}"
                )
                if i >= 9:
                    break

    def get_loop_list(self):
        return [
            (candidate.frame_a, candidate.frame_b)
            for candidate in self.loop_closures
        ]

    def run(self):
        """Run complete loop detection pipeline"""
        print('Loading model...')
        self.load_model()

        if not self.image_paths:
            raise ValueError("image manifest contains no images")

        print(f"Found {len(self.image_paths)} image files")

        self.extract_descriptors()

        self.find_loop_closures()
        self.save_results()
        return self.loop_closures
