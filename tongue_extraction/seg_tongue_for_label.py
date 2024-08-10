from ultralytics.data.annotator import auto_annotate

auto_annotate(data="/data/images", det_model="runs/detect/train/weights/best.pt", sam_model='models/sam_b.pt', output_dir="data_unl/auto_annotate_labels_sam_b")