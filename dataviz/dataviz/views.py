from django.http import HttpResponse
from django.shortcuts import  render
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from io import BytesIO

def home (request):
    return render(request,'index.html')

def upload(request):
    if "csvFile" in request.FILES:
        csvFileObject = request.FILES["csvFile"]
        csvFile = BytesIO(csvFileObject.read())
        data = pd.read_csv(csvFile)
        print(data.value_counts())
        return HttpResponse("File uploaded")
    else:
        return HttpResponse("No File selected")