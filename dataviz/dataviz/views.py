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

uuid_file_dict = {}
graph_list = ["Line Graph", "Bar Graph","Scatter Plot","HeXBin Plot"]
graph_function_dict = {
    "Line Graph": graphs.get_line_graph, 
    "Bar Graph": graphs.get_bar_graph,
    "Scatter Plot" : graphs.get_scatter_plot,
    "HeXBin Plot" : graphs.get_hexbin_plot}

# Home Function


def home(request):
    return render(request, 'index.html')

# Upload CSV File Function


def upload(request):
    # Checking if the request contains a csv file
    if "csvFile" in request.FILES:
        csv_file_object = request.FILES["csvFile"]
        csv_file = BytesIO(csv_file_object.read())

        # Creating a uuid so that user dont have to upload his csv files to visualize multiple graphs
        csv_uuid = shortuuid.ShortUUID().random(length=10)

        # Storing csv file in dictionary to use it for further requests
        uuid_file_dict[csv_uuid] = csv_file

        return redirect("/"+csv_uuid)
    else:
        return HttpResponse("No File selected")


def load_data(uuid_key):
    if uuid_key in uuid_file_dict:
        csv_file = uuid_file_dict[uuid_key]
        csv_file.seek(0)
        data = pd.read_csv(csv_file)
    else:
        data = None

    return data


def show_data(request, csv_uuid):
    data = load_data(csv_uuid)
    if data is not None:
        param = {"data_columns": list(data.columns), "data_values": list(
            data.head().values), "graph_list": graph_list}
        param["plot"] = None
        if (request.method == 'GET'):
            graph_type = request.GET.get("graph")
            y_column = request.GET.get("ycolumn")
            x_column = request.GET.get("xcolumn")
            if (graph_type is not None):
                plot = get_graph(data, graph_type, x_column, y_column)
                param["plot"] = plot
        return render(request, 'show_data.html', param)
    else:
        return HttpResponse("Page Not Found")


def get_graph(data, graph_type, x_column, y_column):

    plot = None
    
    if graph_type in graph_function_dict:
        print("I am in ")
        plot = graph_function_dict[graph_type](
            data=data, x_column=x_column, y_column=y_column)
    
    return plot
