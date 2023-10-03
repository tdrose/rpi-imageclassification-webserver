from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

import json
import base64
from .torchdetection import cap, net, get_imageprediction, imagenet_1000_classes
import io
from PIL import Image
import numpy as np

# Create your views here.
def index(request):

    return render(request, 'app/index.html', {})

@csrf_exempt
def get_prediction(request):
    image, objectclass, percentage = get_imageprediction(net, cap, imagenet_1000_classes)

    pil_image = Image.fromarray(image)

    # Encode the PIL Image in JPEG format
    with io.BytesIO() as output:
        pil_image.save(output, format="JPEG")
        image_data = output.getvalue()

    base64_image = base64.b64encode(image_data).decode('utf-8')

    # Additional text
    additional_text = f"Prediction: {objectclass} ({percentage: .1%})"

    # Construct a dictionary with image and text
    response_data = {
        "image": f"data:image/jpeg;base64,{base64_image}",
        "text": additional_text
    }
    #response_json = json.dumps(response_data)
    #return HttpResponse(response_json, content_type='application/json')

    return JsonResponse(response_data)