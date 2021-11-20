import easyocr
import cv2

# Api doc: https://www.jaided.ai/easyocr/documentation/

ine_low_resolution = cv2.imread("id/ine.jpg")
ine_higher_resolution = cv2.imread("id/ine.png")
pasaporte_higher_resolution = cv2.imread("id/passporte-mx.png")

reader = easyocr.Reader(["es","es"], gpu=False, download_enabled=False) # this needs to run only once to load the model into memory
# Parameters
# lang_list (list) - list of language code you want to recognize, for example ['ch_sim','en']. List of supported language code is here.
# gpu (bool, string, default = True) - enable GPU
# model_storage_directory (string, default = None) - Path to directory for model data. If not specified, models will be read from a directory as defined by the environment variable EASYOCR_MODULE_PATH (preferred), MODULE_PATH (if defined), or ~/.EasyOCR/.
# download_enabled (bool, default = True) - enable download if EasyOCR is not able locate model files
# user_network_directory (bool, default = None) - Path to user-defined recognition network. If not specified, models will be read from MODULE_PATH + '/user_network' (~/.EasyOCR/user_network).
# recog_network (string, default = 'standard') - Instead of standard mode, you can choose your own recognition network --- Tutorial to be written about this.
# detector (bool, default = True) - load detection model into memory
# recognizer (bool, default = True) - load recognition model into memory

# Procesamiento OCR de INE
pine_low_resolution = reader.readtext(ine_low_resolution,detail = 0, paragraph=False)

pine_higher_resolution = reader.readtext(ine_higher_resolution,detail = 0, paragraph=False)

# Procesamiento OCR de Pasaporte
ppasaporte_higher_resolution = reader.readtext(pasaporte_higher_resolution,detail = 0, paragraph=False)

# Parametros 1: General
# image (string, numpy array, byte) - Input image
# decoder (string, default = 'greedy') - options are 'greedy', 'beamsearch' and 'wordbeamsearch'.
# beamWidth (int, default = 5) - How many beam to keep when decoder = 'beamsearch' or 'wordbeamsearch'
# batch_size (int, default = 1) - batch_size>1 will make EasyOCR faster but use more memory
# workers (int, default = 0) - Number thread used in of dataloader
# allowlist (string) - Force EasyOCR to recognize only subset of characters. Useful for specific problem (E.g. license plate, etc.)
# blocklist (string) - Block subset of character. This argument will be ignored if allowlist is given.
# detail (int, default = 1) - Set this to 0 for simple output
# paragraph (bool, default = False) - Combine result into paragraph
# min_size (int, default = 10) - Filter text box smaller than minimum value in pixel
# rotation_info (list, default = None) - Allow EasyOCR to rotate each text box and return the one with the best confident score. Eligible values are 90, 180 and 270. For example, try [90, 180 ,270] for all possible text orientations.

# 1. INE Baja resolucion
print("# 1. INE Baja resolucion")
print (pine_low_resolution)

# 2. INE Alta resolucion
print("# 2. INE Alta resolucion")
print (pine_higher_resolution)

# 3. PASAPORTE Alta resolucion
print("# 3. Pasaporte MX")
print (ppasaporte_higher_resolution)