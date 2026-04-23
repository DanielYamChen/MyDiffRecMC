import cv2
import OpenEXR
import Imath
import numpy as np

input_path = "/home/bohsun/UW_Madison/Research/DiffPhysCam_Data/NovelViewSynthesis_Data/RealScene01/envmaps/envmap_180_045_nvDiffRec_exp.hdr"
output_path = "/home/bohsun/UW_Madison/Research/DiffPhysCam_Data/NovelViewSynthesis_Data/SimExperiment/envmaps/envmap_090_045_nvDiffRec_exp.exr"

img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
assert (img is not None), f"Failed to read image: {input_path}"

img = img.astype(np.float32)
assert (img.shape[2] == 3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img[:, 188 : 195, :] = img[:, 60:67, :]
img[:, 60:67, :] = img[:, 0:7, :]


header = OpenEXR.Header(img.shape[1], img.shape[0])
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
header["channels"] = {
    "R": Imath.Channel(FLOAT),
    "G": Imath.Channel(FLOAT),
    "B": Imath.Channel(FLOAT),
}

exr = OpenEXR.OutputFile(output_path, header)
## OpenCV RGB -> EXR RGB
exr.writePixels({
    "R": np.ascontiguousarray(img[:, :, 0]).tobytes(),
    "G": np.ascontiguousarray(img[:, :, 1]).tobytes(),
    "B": np.ascontiguousarray(img[:, :, 2]).tobytes(),
})
exr.close()
