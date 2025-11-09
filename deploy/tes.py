from pyzbar.pyzbar import ZBarSymbol, ZBarLibrary, decode
from PIL import Image
import os

# --- Step 1: Load ZBar DLL manually using constructor ---
dll_path = r"C:\Users\My pc\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyzbar\libzbar-64.dll"  # <-- Update path to your actual DLL location

# Explicitly load the library
libzbar = ZBarLibrary(dll_path)

# --- Step 2: Test decoding ---
image_path = r":\Semester 8\CV\Assignment\Halal Food\barcode\test\images\-_-_mfnr_jpg.rf.e2923fc4441ba1c6e882697615c97ad9,jpg"  # <-- update with your actual image path
image = Image.open(image_path)

# Decode using the explicitly loaded library
decoded_objects = decode(image, symbols=[ZBarSymbol.EAN13, ZBarSymbol.CODE128, ZBarSymbol.QRCODE], libzbar=libzbar)

# --- Step 3: Print results ---
if decoded_objects:
    for obj in decoded_objects:
        print("Type:", obj.type)
        print("Data:", obj.data.decode("utf-8"))
else:
    print("No barcode found.")
