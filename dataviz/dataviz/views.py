import base64
from pickle import GET
from django.http import HttpResponse
from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import shortuuid
from . import graphs
import datetime
import os
import glob

TIMEOUT_TIME = 300

uuid_file_dict = {}
uuid_file_last_modified_datetime_dict = {}
sample_file__list = []
sample_files_image_path_dict = {}


sample_files_path = "./static/sample_datasets"

#Loading Sample Datasets
files = glob.glob(sample_files_path + "/*.csv")
#print(files)
for file in files:
    with open(file,"rb") as csvfile_object:
        csvfile = BytesIO(csvfile_object.read())
        filename = file.split('/')[-1].split('.')[0]
        sample_file__list.append(filename)
        sample_files_image_path_dict[filename] = "sample_datasets_images/"+filename+".jpg"
        uuid_file_dict[filename] = csvfile
        uuid_file_last_modified_datetime_dict[filename] = datetime.datetime.now()

#print(sample_file__list)

graph_list = ["Line Graph", "Bar Graph","Scatter Plot","HeXBin Plot","Stem Plot", "2D Histogram", "TriPlot"]
# Which matplotlib style sheet to use
style_list = ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', 
            '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 
            'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 
            'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 
            'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 
            'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 
            'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 
            'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid']
graph_function_dict = {
    "Line Graph": graphs.get_line_graph, 
    "Bar Graph": graphs.get_bar_graph,
    "Scatter Plot" : graphs.get_scatter_plot,
    "HeXBin Plot" : graphs.get_hexbin_plot,
    "Stem Plot" : graphs.get_stem_plot,
    "2D Histogram" : graphs.get_hist2d_plot,
    "TriPlot" : graphs.get_triplot_plot}


def delete_unused_files_decorator(func):
    def inner1(*args, **kwargs):
        nowTime = datetime.datetime.now()
        keysToDelete = []
        for key in uuid_file_dict:
            fileTime = uuid_file_last_modified_datetime_dict[key]
            timediff = nowTime-fileTime
            if(timediff.seconds > TIMEOUT_TIME and key not in sample_file__list):
                keysToDelete.append(key)

        for key in keysToDelete:
            uuid_file_dict.pop(key)
            uuid_file_last_modified_datetime_dict.pop(key)
        return func(*args, **kwargs)
    return inner1

# Home Function
@delete_unused_files_decorator
def home(request):
    param = {}
    param["sample_dataset_list"] = sample_file__list
    param["sample_files_image_path_dict"] = sample_files_image_path_dict
    return render(request, 'index.html',param)

# Upload CSV File Function
@delete_unused_files_decorator
def upload(request):
    # Checking if the request contains a csv file
    if "csvFile" in request.FILES:
        csv_file_object = request.FILES["csvFile"]
        csv_file = BytesIO(csv_file_object.read())

        # Creating a uuid so that user dont have to upload his csv files to visualize multiple graphs
        csv_uuid = shortuuid.ShortUUID().random(length=10)

        # Storing csv file in dictionary to use it for further requests
        uuid_file_dict[csv_uuid] = csv_file
        uuid_file_last_modified_datetime_dict[csv_uuid] = datetime.datetime.now()

        return redirect("/"+csv_uuid)
    else:
        return HttpResponse("No File selected")


@delete_unused_files_decorator
def show_data(request, csv_uuid):
    data = load_data(csv_uuid)
    if data is not None:
        param = {"data_columns": list(data.columns), "data_values": list(
            data.head().values), "graph_list": graph_list}
        param["plot"] = None
        param["plot_style_list"] = style_list
        if (request.method == 'GET'):
            graph_type = request.GET.get("graph")
            y_column = request.GET.get("ycolumn")
            x_column = request.GET.get("xcolumn")
            # getting style param
            style = request.GET.get("style")
            if (graph_type is not None):
                plot = get_graph(data, graph_type, x_column, y_column, style)
                param["plot"] = plot
        return render(request, 'show_data.html', param)
    else:
        return redirect("/")


def load_data(uuid_key):
    if uuid_key in uuid_file_dict:
        csv_file = uuid_file_dict[uuid_key]
        uuid_file_last_modified_datetime_dict[uuid_key] = datetime.datetime.now()
        csv_file.seek(0)
        data = pd.read_csv(csv_file)
    else:
        data = None

    return data

def get_graph(data, graph_type, x_column, y_column, style):

    plot = None
    if graph_type in graph_function_dict:
        plot = graph_function_dict[graph_type](
            data=data, x_column=x_column, y_column=y_column, style=style)
    
    return plot