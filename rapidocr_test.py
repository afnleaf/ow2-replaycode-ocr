from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()

img_path = '/app/output/char_0.png'
result, elapse = engine(img_path)
print(result)
print(elapse)
