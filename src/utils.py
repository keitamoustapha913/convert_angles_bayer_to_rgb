import os
from glob import glob
from typing import List
import shutil
import cv2
import polanalyser as pa
import numpy as np
import time





def main():
    # Read polarization image
    filepath = "/home/theyeq-admin/Documents/convert_angles_bayer_to_rgb/images/raw_samples/from_PolarizedAngles_0d_45d_90d_135d_BayerRG8_full_channel__0a22a3f2-6b5a-11ee-8c8e-00044bec23a2_to_png_with_cv2.png"
    img_raw = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    print(img_raw.shape)

    img_raw_reshaped = np.reshape(img_raw, (2048, -1))
    print( "img_raw_reshaped shape : " , img_raw_reshaped.shape)

    # Demosaicing
    img_000, img_045, img_090, img_135 = pa.demosaicing(img_raw_reshaped, pa.COLOR_PolarRGB) # COLOR_PolarRGB COLOR_PolarMono

    print(f"Export demosaicing images : {filepath}")

    name, ext = os.path.splitext(filepath)
    cv2.imwrite(f"{name}-000{ext}", img_000)
    cv2.imwrite(f"{name}-045{ext}", img_045)
    cv2.imwrite(f"{name}-090{ext}", img_090)
    cv2.imwrite(f"{name}-135{ext}", img_135)

def my_demosaic():
    # Read polarization image
    filepath = "/home/theyeq-admin/Documents/convert_angles_bayer_to_rgb/images/raw_samples/polar_0a22a3f2-6b5a-11ee-8c8e-00044bec23a2_0deg_raw.png"
    img_raw = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    print(img_raw.shape)

    name, ext = os.path.splitext(filepath)

    code_BG2BGR = getattr(cv2, f"COLOR_BayerBG2BGR")
    img_debayer_BG2BGR = cv2.cvtColor(img_raw, code_BG2BGR)
    print("img_debayer_BG2BGR.shape : ", img_debayer_BG2BGR.shape)
    cv2.imwrite(f"{name}-000_debayer_BG2BGR{ext}", img_debayer_BG2BGR)

    code_BG2RGB = getattr(cv2, f"COLOR_BayerBG2RGB")
    img_debayer_BG2RGB = cv2.cvtColor(img_raw, code_BG2RGB)
    print("img_debayer_BG2RGB.shape : ", img_debayer_BG2RGB.shape)
    cv2.imwrite(f"{name}-000_debayer_BG2RGB{ext}", img_debayer_BG2RGB)


def polarized_angles_bayer_rg8_to_bgr8_from_path(img_polarized_angles_bayer_rg8_path: str = "PolarizedAngles_0d_45d_90d_135d_BayerRG8.png" ):
    """
        PolarizedAngles_0d_45d_90d_135d_BayerRG8 to PolarizedAngles_0d_BGR8
                                                    PolarizedAngles_45d_BGR8
                                                    PolarizedAngles_90d_BGR8
                                                    PolarizedAngles_135d_BGR8
    """
    img_raw = cv2.imread(img_polarized_angles_bayer_rg8_path, cv2.IMREAD_UNCHANGED)
    img_raw_angles_split_list = cv2.split(img_raw) # [ img_000_raw, img_045_raw , img_090_raw, img_135_raw]
    angles_names_list = ["0d", "45d", "90d", "135d"]
    code_bg2bgr = getattr(cv2, f"COLOR_BayerBG2BGR")

    #for img_raw_angles_split in img_raw_angles_split_list:
    #    img_debayer_BG2BGR = cv2.cvtColor(img_raw_angles_split, code_bg2bgr)
    #    print("img_debayer_BG2BGR.shape : ", img_debayer_BG2BGR.shape)

    img_angles_debayer_bg2bgr_list = [  cv2.cvtColor(img_raw_angles_split, code_bg2bgr) for img_raw_angles_split in img_raw_angles_split_list]

    print("img_angles_debayer_bg2bgr_list length : ", len(img_angles_debayer_bg2bgr_list))

    name, ext = os.path.splitext(img_polarized_angles_bayer_rg8_path)
    for i, (angle_name, img_angle_bgr8) in enumerate(zip(angles_names_list, img_angles_debayer_bg2bgr_list)):
        cv2.imwrite(f"{name}_{angle_name}_BG2BGR{ext}", img_angle_bgr8)
        print( f" {i} : {name}_{angle_name}_BG2BGR{ext} shape : {img_angle_bgr8.shape}")

def polarized_angles_bayer_rg8_to_bgr8(img_polarized_angles_bayer_rg8: np.ndarray) -> List[np.ndarray]:
    """
        PolarizedAngles_0d_45d_90d_135d_BayerRG8 to PolarizedAngles_0d_BGR8
                                                    PolarizedAngles_45d_BGR8
                                                    PolarizedAngles_90d_BGR8
                                                    PolarizedAngles_135d_BGR8
    """
    if (img_polarized_angles_bayer_rg8.ndim != 2) and (img_polarized_angles_bayer_rg8.shape[2] != 4) :
        raise ValueError(f"The dimension of the input image must be 2 and shape must be (height, width, 4) , not {img_polarized_angles_bayer_rg8.ndim} {img_polarized_angles_bayer_rg8.shape}")
    
    img_raw_angles_split_list = cv2.split(img_polarized_angles_bayer_rg8) # [ img_000_raw, img_045_raw , img_090_raw, img_135_raw]
    code_bg2bgr = getattr(cv2, "COLOR_BayerBG2BGR")

    polarized_angles_0d_bgr8, polarized_angles_45d_bgr8, polarized_angles_90d_bgr8, polarized_angles_135d_bgr8 = [  cv2.cvtColor(img_raw_angles_split, code_bg2bgr) for img_raw_angles_split in img_raw_angles_split_list]

    return [polarized_angles_0d_bgr8, polarized_angles_45d_bgr8, polarized_angles_90d_bgr8, polarized_angles_135d_bgr8]

def save_polarized_angles(polarized_angles_xx_bgr8_imgs_list: List[np.ndarray], directory_path: str = ""):
    polarized_angle_img_filenames_list = ["polarized_angles_0d_bgr8", "polarized_angles_45d_bgr8", "polarized_angles_90d_bgr8", "polarized_angles_135d_bgr8"]
    for i, (polarized_angle_img_filename, polarized_angles_xx_bgr8_img) in enumerate(zip(polarized_angle_img_filenames_list, polarized_angles_xx_bgr8_imgs_list)):
        polarized_angles_xx_bgr8_img_path = os.path.join( directory_path,  f"{polarized_angle_img_filename}.png")
        cv2.imwrite( polarized_angles_xx_bgr8_img_path, polarized_angles_xx_bgr8_img)
        print( f" Saved {i} : {polarized_angles_xx_bgr8_img_path} shape : {polarized_angles_xx_bgr8_img.shape}")


def convert_and_save_polarized_angles_to_bgr8(img_polarized_angles_bayer_rg8: np.ndarray , directory_path: str = "" ) -> None:
    polarized_angles_xx_bgr8_imgs_list = polarized_angles_bayer_rg8_to_bgr8(img_polarized_angles_bayer_rg8 = img_polarized_angles_bayer_rg8)
    save_polarized_angles(polarized_angles_xx_bgr8_imgs_list = polarized_angles_xx_bgr8_imgs_list, directory_path = directory_path)

def convert_and_save_polarized_angles_to_bgr8_from_path(img_polarized_angles_bayer_rg8_filepath: str = "" , directory_path: str = "" ) -> None:
    if not os.path.exists(img_polarized_angles_bayer_rg8_filepath):
        return None
    img_polarized_angles_bayer_rg8 = cv2.imread(img_polarized_angles_bayer_rg8_filepath , cv2.IMREAD_UNCHANGED)
    polarized_angles_xx_bgr8_imgs_list = polarized_angles_bayer_rg8_to_bgr8(img_polarized_angles_bayer_rg8 = img_polarized_angles_bayer_rg8)
    save_polarized_angles(polarized_angles_xx_bgr8_imgs_list = polarized_angles_xx_bgr8_imgs_list, directory_path = directory_path)


def extract_img_polarized_angles_bayer_rg8_path(zipped_compressed_imgs_path: str = "", unzipped_compressed_dir: str = "" ):
    print(zipped_compressed_imgs_path)
    print(unzipped_compressed_dir)
    if not os.path.exists(zipped_compressed_imgs_path) or (unzipped_compressed_dir == ""):
        return None

    basename = os.path.splitext( os.path.basename(zipped_compressed_imgs_path))[0]
    #print(f"basename {basename}")
    extraction_directory = os.path.join( unzipped_compressed_dir, f"{basename}")
    if not os.path.exists(extraction_directory):
        os.makedirs(extraction_directory)
    print( " extraction_directory : ", extraction_directory)
    archive_format = "zip"
    shutil.unpack_archive(zipped_compressed_imgs_path, extraction_directory, archive_format)

    img_polarized_angles_bayer_rg8_path = glob( os.path.join(extraction_directory, "from_PolarizedAngles_**.png" ) , recursive = False)[0]
    print(f" Found img_polarized_angles_bayer_rg8_path : {img_polarized_angles_bayer_rg8_path}")

    convert_and_save_polarized_angles_to_bgr8_from_path( img_polarized_angles_bayer_rg8_filepath = img_polarized_angles_bayer_rg8_path , directory_path = extraction_directory)


def adjust_gamma(image, gamma):
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    table = (255.0 * (np.linspace(0, 1, 256) ** gamma)).astype(np.uint8)
    return cv2.LUT(image_u8, table)


def generate_colormap(color0, color1):
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[:128] = np.linspace(1, 0, 128)[..., None] * np.array(color0)
    colormap[128:] = np.linspace(0, 1, 128)[..., None] * np.array(color1)
    return np.clip(colormap, 0, 255)

def normalise_img_to_uint8(img, target_type_min, target_type_max, target_type):
    """
    Normalise img to uint8 with value between target_type_min to target_type_max
    """
    imin = img.min()
    imax = img.max()
    a = ( target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b ).astype(target_type)
    return new_img

def calculate_stokes_and_save(img_polarized_angles_bayer_rg8_path:str = "from_PolarizedAngles_0d_45d_90d_135d_BayerRG8.png"):
    img_polarized_angles_bayer_rg8 = cv2.imread(img_polarized_angles_bayer_rg8_path , cv2.IMREAD_UNCHANGED)
    print("img_polarized_angles_bayer_rg8 shape : ", img_polarized_angles_bayer_rg8.shape)
    polarized_angles_list = polarized_angles_bayer_rg8_to_bgr8(img_polarized_angles_bayer_rg8 = img_polarized_angles_bayer_rg8)
    #polarized_angles_0d_bgr8, polarized_angles_45d_bgr8, polarized_angles_90d_bgr8, polarized_angles_135d_bgr8 = polarized_angles_list
    
    # Convert BGR to Gray
    code_bgr2gray = getattr(cv2, "COLOR_BGR2GRAY")
    polarized_angles_list_gray = [  cv2.cvtColor(polarized_angle, code_bgr2gray) for polarized_angle in polarized_angles_list]


    # Get the directory name from the specified path 
    directory_path = os.path.dirname(img_polarized_angles_bayer_rg8_path) 
    save_polarized_angles(polarized_angles_xx_bgr8_imgs_list = polarized_angles_list_gray, directory_path = directory_path)
    
    # Calculate Stokes vector from intensity images and polarizer angles
    polarized_angles_list_u32 = [  np.int32(polarized_angle) for polarized_angle in polarized_angles_list_gray]

    polarized_angles_list_used = polarized_angles_list_gray
    start_time = time.time()
    aolp_result = calculate_aolp(*polarized_angles_list_used)
    print("\n aolp_result ", aolp_result)
    print("\n aolp_result shape", aolp_result.shape, " min : ", aolp_result.min(), " max : ", aolp_result.max())

    dolp_result = calculate_dolp(*polarized_angles_list_used)
    print("\n dolp_result ", dolp_result)
    print("\n dolp_result shape", dolp_result.shape, " min : ", dolp_result.min(), " max : ", dolp_result.max())

    aolp_normalized = calculate_aolp_norm(*polarized_angles_list_used)
    print("\n aolp_normalized ", aolp_normalized)
    print("\n aolp_normalized shape", aolp_normalized.shape, " min : ", aolp_normalized.min(), " max : ", aolp_normalized.max())

    dolp_normalized = calculate_dolp_norm(*polarized_angles_list_used)
    print("\n dolp_normalized ", dolp_normalized)
    print("\n dolp_normalized shape", dolp_normalized.shape, " min : ", dolp_normalized.min(), " max : ", dolp_normalized.max())

    lpi_normalized = calculate_intensity_norm(*polarized_angles_list_used)
    print("\n lpi_normalized ", lpi_normalized)
    print("\n lpi_normalized shape", lpi_normalized.shape, " min : ", lpi_normalized.min(), " max : ", lpi_normalized.max())

    lpi_dolp_aolp_norm = cv2.merge([lpi_normalized, dolp_normalized, aolp_normalized])
    print("\n lpi_dolp_aolp_norm ", lpi_dolp_aolp_norm)
    print("\n lpi_dolp_aolp_norm shape", lpi_dolp_aolp_norm.shape, " min : ", lpi_dolp_aolp_norm.min(), " max : ", lpi_dolp_aolp_norm.max())
    end_time = time.time()
    print(" End time : ", end_time - start_time)



    stokesI = polarized_angles_list_u32[0] + polarized_angles_list_u32[2]
    stokesQ = polarized_angles_list_u32[0] - polarized_angles_list_u32[2]
    stokesU = polarized_angles_list_u32[1] + polarized_angles_list_u32[3]
    img_stokes = [ stokesI , stokesQ , stokesU]
    print("Calculates : IMG STOKES DONE !")

    # Stokes to parameters (s0, s1, s2, Intensity(s0) DoLP, AoLP)
    img_s0 = stokesI # normalise_img_to_uint8(stokesI, 0, 255, np.uint8)
    img_s1 = stokesQ # normalise_img_to_uint8(stokesQ, 0, 255, np.uint8)
    img_s2 = stokesU # normalise_img_to_uint8(stokesU, 0, 255, np.uint8)
    img_intensity_raw = np.sqrt( np.add( np.square(stokesQ) , np.square(stokesU) ) )
    img_intensity = normalise_img_to_uint8(img_intensity_raw.astype(np.float32), 0, 255, np.uint8)
    img_dolp_raw = np.divide( img_intensity_raw.astype(np.float32), stokesI, where = stokesI>0 )  # [0, 1]
    img_dolp = normalise_img_to_uint8(img_dolp_raw, 0, 1, np.uint8)
    img_aolp_raw = (0.5 * ( np.arctan2(stokesU, stokesQ)))  # [0, pi]
    img_aolp = normalise_img_to_uint8(img_aolp_raw, 0, np.pi, np.uint8)
    

    # Export images
    name, _ext = os.path.splitext(img_polarized_angles_bayer_rg8_path)
    cv2.imwrite(f"{name}_s0.png", img_s0)
    cv2.imwrite(f"{name}_s1.png", img_s1)
    cv2.imwrite(f"{name}_s2.png", img_s2)
    cv2.imwrite(f"{name}_intensity.png", img_intensity)
    cv2.imwrite(f"{name}_DoLP.png", img_dolp)
    cv2.imwrite(f"{name}_AoLP.png", img_aolp)

    cv2.imwrite(f"{name}_dolp_result.png", dolp_result)
    cv2.imwrite(f"{name}_aolp_result.png", aolp_result)
    cv2.imwrite(f"{name}_dolp_normalized.png", dolp_normalized)
    cv2.imwrite(f"{name}_aolp_normalized.png", aolp_normalized)
    cv2.imwrite(f"{name}_lpi_normalized.png", lpi_normalized)
    cv2.imwrite(f"{name}_lpi_dolp_aolp_norm.png", lpi_dolp_aolp_norm)
    

    print("Calculates : IMG (s0, s1, s2, Intensity(s0) DoLP, AoLP) DONE !")
    # Custom colormap (Positive -> Green, Negative -> Red)
    custom_colormap = generate_colormap((0, 0, 255), (0, 255, 0))

    # Apply colormap or adjust the brightness to export images
    img_s0_u8 = pa.applyColorMap(img_s0, "viridis", vmin=0, vmax=np.max(img_s0))
    img_s1_u8 = pa.applyColorMap(img_s1, "viridis", vmin=0, vmax=np.max(img_s1))  # normalized by s0
    img_s2_u8 = pa.applyColorMap(img_s2 , "viridis", vmin=0, vmax=np.max(img_s2))  # normalized by s0
    img_intensity_u8 = adjust_gamma(img_intensity * 0.5, gamma=(1 / 2.2))
    img_dolp_u8 = np.clip(img_dolp * 255, 0, 255).astype(np.uint8)
    img_aolp_u8 = pa.applyColorToAoLP(img_aolp)  # Hue = AoLP, Saturation = 1, Value = 1
    img_aolp_s_u8 = pa.applyColorToAoLP(img_aolp, saturation=img_dolp)  # Hue = AoLP, Saturation = DoLP, Value = 1
    img_aolp_v_u8 = pa.applyColorToAoLP(img_aolp, value=img_dolp)  # Hue = AoLP, Saturation = 1, Value = DoLP

    
    cv2.imwrite(f"{name}_s0_u8.png", img_s0_u8)
    cv2.imwrite(f"{name}_s1_u8.png", img_s1_u8)
    cv2.imwrite(f"{name}_s2_u8.png", img_s2_u8)
    cv2.imwrite(f"{name}_intensity_u8.png", img_intensity_u8)
    cv2.imwrite(f"{name}_DoLP_u8.png", img_dolp_u8)
    cv2.imwrite(f"{name}_AoLP_u8.png", img_aolp_u8)
    cv2.imwrite(f"{name}_AoLP_s_u8.png", img_aolp_s_u8)
    cv2.imwrite(f"{name}_AoLP_v_u8.png", img_aolp_v_u8)

    print("calculate_stokes_and_save DONE !")

def calculate_aolp(image_0, image_45, image_90, image_135):
    """
    # Example usage:
    # Assuming you have images_0, images_45, images_90, and images_135
    # Replace these placeholders with actual image data

    # aolp_result = calculate_aolp(images_0, images_45, images_90, images_135)
    # print(aolp_result)
    """
    # Convert the images to numpy arrays for easier manipulation
    image_0 = np.array(image_0)
    image_45 = np.array(image_45)
    image_90 = np.array(image_90)
    image_135 = np.array(image_135)

    # Calculate Stokes parameters
    Q = image_0 - image_90
    U = image_45 - image_135

    # Calculate angle of linear polarization (AoLP)
    aolp = 0.5 * np.arctan2(U, Q)

    # Convert angle from radians to degrees
    aolp_degrees = np.degrees(aolp)

    return aolp

def calculate_dolp(image_0, image_45, image_90, image_135):
    """
    # Example usage:
    # Assuming you have images_0, images_45, images_90, and images_135
    # Replace these placeholders with actual image data

    # dolp_result = calculate_dolp(images_0, images_45, images_90, images_135)
    # print(dolp_result)
    """
    # Convert the images to numpy arrays for easier manipulation
    image_0 = np.array(image_0)
    image_45 = np.array(image_45)
    image_90 = np.array(image_90)
    image_135 = np.array(image_135)

    # Perform necessary calibration or normalization based on the sensor documentation

    # Assuming the images are properly calibrated and normalized, calculate DoLP
    numerator = np.sqrt((image_0 - image_90)**2 + (image_45 - image_135)**2)
    denominator = image_0 + image_45 + image_90 + image_135

    # Avoid division by zero
    #dolp = np.where(denominator != 0, numerator / denominator, 0)
    dolp = np.divide(numerator , denominator, where=denominator!=0)

    return dolp

def calculate_aolp_norm(image_0, image_45, image_90, image_135):
    # Convert the images to numpy arrays for easier manipulation
    image_0 = np.array(image_0, dtype=np.float32)
    image_45 = np.array(image_45, dtype=np.float32)
    image_90 = np.array(image_90, dtype=np.float32)
    image_135 = np.array(image_135, dtype=np.float32)

    # Calculate Stokes parameters
    Q = image_0 - image_90
    U = image_45 - image_135

    # Calculate Angle of Linear Polarization (AoLP)
    aolp_radians = 0.5 * np.arctan2(U, Q)
    aolp_degrees = np.degrees(aolp_radians)

    # Normalize AoLP to [0, 255] for HSV representation
    aolp_normalized = ((aolp_degrees + 180) * 255 / 360).astype(np.uint8)
    print(" aolp_normalized.shape : ", aolp_normalized.shape)
    return aolp_normalized

def create_hsv_aolp_image(aolp_normalized):
    # Create an HSV image with saturation and value set to maximum
    hsv_image = np.zeros((aolp_normalized.shape[0], aolp_normalized.shape[1], 3), dtype=np.uint8)
    hsv_image[:, :, 0] = aolp_normalized  # Hue corresponds to AoLP
    hsv_image[:, :, 1:] = 255  # Saturation and Value set to maximum

    # Convert HSV to BGR for visualization
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return bgr_image

def calculate_dolp_norm(image_0, image_45, image_90, image_135):
    # Convert the images to numpy arrays for easier manipulation
    image_0 = np.array(image_0, dtype=np.float32)
    image_45 = np.array(image_45, dtype=np.float32)
    image_90 = np.array(image_90, dtype=np.float32)
    image_135 = np.array(image_135, dtype=np.float32)

    # Calculate the Stokes parameters
    Q = image_0 - image_90
    U = image_45 - image_135

    # Calculate Degree of Linear Polarization (DoLP)
    #dolp = np.sqrt(Q**2 + U**2) / (image_0 + image_45 + image_90 + image_135)
    numerator = np.sqrt(Q**2 + U**2) 
    denominator = image_0 + image_45 + image_90 + image_135
    # Avoid division by zero
    #dolp = np.where(denominator != 0, numerator / denominator, 0)
    dolp = np.divide(numerator , denominator, where=denominator!=0)


    # Normalize DoLP to [0, 255] for HSV representation
    dolp_normalized = (dolp * 255).astype(np.uint8)

    return dolp_normalized

def calculate_intensity_norm(image_0, image_45, image_90, image_135):
    # Convert the images to numpy arrays for easier manipulation
    image_0 = np.array(image_0, dtype=np.float32)
    image_45 = np.array(image_45, dtype=np.float32)
    image_90 = np.array(image_90, dtype=np.float32)
    image_135 = np.array(image_135, dtype=np.float32)

    # Calculate Stokes parameters
    Q = image_0 - image_90
    U = image_45 - image_135

    # Calculate Intensity of Linear Polarization (ILP)
    ilp = np.sqrt(Q**2 + U**2)

    # Normalize ILP to [0, 255] for HSV representation
    ilp_normalized = (ilp * 255 / np.max(ilp)).astype(np.uint8)

    return ilp_normalized


def create_hsv_image(dolp_normalized):
    # Create an HSV image with saturation and value set to maximum
    hsv_image = np.zeros((dolp_normalized.shape[0], dolp_normalized.shape[1], 3), dtype=np.uint8)
    hsv_image[:, :, 0] = dolp_normalized  # Hue corresponds to DoLP
    hsv_image[:, :, 1:] = 255  # Saturation and Value set to maximum

    # Convert HSV to BGR for visualization
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return bgr_image

def create_polarized_image(image_0, image_45, image_90, image_135, target_angle_degrees=30):
    """
        Generate a polarized image at 30 degrees from polarimetric images.

        Parameters:
        - image_0: Array, polarimetric image at 0 degrees.
        - image_45: Array, polarimetric image at 45 degrees.
        - image_90: Array, polarimetric image at 90 degrees.
        - image_135: Array, polarimetric image at 135 degrees.

        Returns:
        - polarized_image: Array, the resulting polarized image at 30 degrees.
    """
    
    # Convert the images to numpy arrays for easier manipulation
    image_0 = np.array(image_0, dtype=np.float32)
    image_45 = np.array(image_45, dtype=np.float32)
    image_90 = np.array(image_90, dtype=np.float32)
    image_135 = np.array(image_135, dtype=np.float32)

    # Calculate the interpolation weights for the target angle
    weight_0 = np.cos(np.radians(target_angle_degrees))**2
    weight_45 = 0.5 * np.sin(np.radians(2 * target_angle_degrees))
    weight_90 = np.sin(np.radians(target_angle_degrees))**2
    weight_135 = 0.5 * np.sin(np.radians(2 * (90 - target_angle_degrees)))

    # Combine images based on interpolation weights
    polarized_image = (
        weight_0 * image_0 +
        weight_45 * image_45 +
        weight_90 * image_90 +
        weight_135 * image_135
    )

    # Normalize the result to the range [0, 255]
    polarized_image = (polarized_image / np.max(polarized_image) * 255).astype(np.uint8)

    return polarized_image


if __name__ == "__main__":
    #img_raw_samples_path = "/home/theyeq-admin/Documents/convert_angles_bayer_to_rgb/images/raw_samples/dragon.png"
    #img_raw_samples_path = "/home/theyeq-admin/Documents/convert_angles_bayer_to_rgb/images/raw_samples/from_PolarizedAngles_0d_45d_90d_135d_BayerRG8_full_channel__dc15724b-6b5f-11ee-9ade-00044bec23a2_to_png_with_cv2.png"
    #img_dragon_raw = cv2.imread(img_raw_samples_path , cv2.IMREAD_UNCHANGED)

    #print("img_dragon_raw shape : ", img_dragon_raw.shape)

    #main()
    #my_demosaic()
    #img_polarized_angles_bayer_rg8_path = "/home/theyeq-admin/Documents/convert_angles_bayer_to_rgb/images/raw_samples/from_PolarizedAngles_0d_45d_90d_135d_BayerRG8_full_channel__dc15724b-6b5f-11ee-9ade-00044bec23a2_to_png_with_cv2.png"
    #polarized_angles_bayer_rg8_to_bgr8_from_path(img_polarized_angles_bayer_rg8_path = img_polarized_angles_bayer_rg8_path)

    #polarized_angles_list = polarized_angles_bayer_rg8_to_bgr8(img_polarized_angles_bayer_rg8 = img_dragon_raw)
    #img_angles_debayer_bg2bgr_list = [  print( f" shape : {polarized_angles.shape}") for polarized_angles in polarized_angles_list]
    #save_polarized_angles(polarized_angles_xx_bgr8_imgs_list = polarized_angles_list)
    img_polarized_angles_bayer_rg8_path = "/home/theyeq-admin/Documents/convert_angles_bayer_to_rgb/images/unzipped/zip_1ddfdbbc-6b5b-11ee-871f-00044bec23a2/from_PolarizedAngles_0d_45d_90d_135d_BayerRG8_full_channel__1ddfdbbc-6b5b-11ee-871f-00044bec23a2_to_png_with_cv2.png"
    calculate_stokes_and_save(img_polarized_angles_bayer_rg8_path = img_polarized_angles_bayer_rg8_path)

    #img_polarized_angles_bayer_rg8 = cv2.imread(img_polarized_angles_bayer_rg8_path , cv2.IMREAD_UNCHANGED)
    #polarized_angles_list = polarized_angles_bayer_rg8_to_bgr8(img_polarized_angles_bayer_rg8 = img_polarized_angles_bayer_rg8)
    #aolp_normalized = calculate_aolp_norm(*polarized_angles_list)
    #bgr_aolp_image = create_hsv_aolp_image(aolp_normalized)



    

