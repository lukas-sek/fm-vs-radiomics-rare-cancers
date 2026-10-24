import SimpleITK as sitk

# Melanoma-001 / lesion0 from fmcib_CSVs/fmcib_original_Melanoma.csv
image_path = "C:/Users/lukin/Projekte/Thesis/fm-vs-radiomics-rare-cancers/data/worc/Melanoma/Melanoma-038/image.nii.gz"
mask_path = "C:/Users/lukin/Projekte/Thesis/fm-vs-radiomics-rare-cancers/data/worc/Melanoma/Melanoma-038/segmentation_lesion1.nii.gz"
coordX, coordY, coordZ = (
    53.84724549802274,-150.44434017407227,-1139.4741311756272
)

image = sitk.ReadImage(image_path)
mask = sitk.ReadImage(mask_path)

# Map physical coordinates to voxel indices on both volumes
img_idx = image.TransformPhysicalPointToIndex((coordX, coordY, coordZ))
mask_idx = mask.TransformPhysicalPointToIndex((coordX, coordY, coordZ))

# Check bounds
size = image.GetSize()
in_bounds = all(0 <= img_idx[i] < size[i] for i in range(3))

# Check values at centroid
val_in_mask = mask.GetPixel(mask_idx) if in_bounds else None
hu_val = image.GetPixel(img_idx) if in_bounds else None

print(f"In bounds: {in_bounds}")
print(f"Voxel index: {img_idx}")
print(f"Mask value at centroid: {val_in_mask}")  # Must be > 0
print(f"CT HU value at centroid: {hu_val}")  # Air is ~ -1000
