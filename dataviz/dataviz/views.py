from pickle import GET
from django.http import HttpResponse
from django.shortcuts import render, redirect

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib
matplotlib.use('Agg')

import datetime
import os
import glob

from base64 import b64decode
from base64 import b64encode
from io import BytesIO
import shortuuid
from . import graphs

#Time out timer for files to be kept on server
TIMEOUT_TIME = 300

uuid_file_dict = {}
uuid_file_last_modified_datetime_dict = {}
sample_file__list = []
sample_files_image_path_dict = {}

#Path to Sample Datasets 
sample_files_path = "./static/sample_datasets"

#Loading Sample Datasets
files = glob.glob(sample_files_path + "/*.csv")

#File Name condition for windows and linux
flag = None
if( "\\" in files[0]):
    flag = "Windows"
else:
    flag = "Linux"

#Loading Sample Datasets to Dictionary
for file in files:
    with open(file,"rb") as csvfile_object:
        csvfile = BytesIO(csvfile_object.read())
        filename = None
        
        if(flag == "Windows"):
            filename = file.split('\\')[-1].split('.')[0]
        else:
            filename = file.split('/')[-1].split('.')[0]
        sample_file__list.append(filename)
        sample_files_image_path_dict[filename] = "sample_datasets_images/"+filename+".jpeg"
        uuid_file_dict[filename] = csvfile
        uuid_file_last_modified_datetime_dict[filename] = datetime.datetime.now()


#List of Supported Graph Types
graph_list = ["Line Graph", "Bar Graph","Scatter Plot","HeXBin Plot","Stem Plot", "2D Histogram", "TriPlot"]

#Dictionary of functions of graph types 
graph_function_dict = {
    "Line Graph": graphs.get_line_graph, 
    "Bar Graph": graphs.get_bar_graph,
    "Scatter Plot" : graphs.get_scatter_plot,
    "HeXBin Plot" : graphs.get_hexbin_plot,
    "Stem Plot" : graphs.get_stem_plot,
    "2D Histogram" : graphs.get_hist2d_plot,
    "TriPlot" : graphs.get_triplot_plot}

#Graph Descriptions
graph_descrition = {
    "Line Graph" : "It is used to connect different points corresponding to a given base point set. The resulting graph shows the trend of the data and its variation based on the points from the x-axis",
    "Bar Graph": "Similar to a line graph, a bar graph also plots corresponding values for a given base point set. However, as you can see the points are not connected in a chain-like fashion.",
    "Scatter Plot" : "Shows the data as a cartesian coordinate point on the graph instead of columns or lines. It is useful to check for the clustering of data or to find general outliers.",
    "HeXBin Plot" : "Scatter plot works well for a small number of data points but for a huge scale of points, it gets crowded. A Hexbin plot splits the plot area into hexagonal boxes(bins) and uses colour to indicate the frequency of points in those bins.",
    "Stem Plot" : "Very much similar to a Histogram but has a bit better capability of data visualisation. Generally used when the number of data points is relatively low.",
    "2D Histogram" : "A 2D density plot, somewhat similar to the Hexbin plot. It is also used to display the density of points in a specific region in our graph.",
    "TriPlot" : "2D Triangular Plot is used to display the relation between points in the form of triangles. It is quite useful when we need to see how the data points interact with each other in form of triplets instead of pairs."
}

#Decorator for checking and deleting files that are not accessed recently
def delete_unused_files_decorator(func):
    def inner1(*args, **kwargs):
        """
        Traverses the list of all files stored on server and deletes the files which are not accessed in TIMEOUT TIME seconds
        before the functionaly of any function starts
    
        Parameters:
        function: Function with below functionaly and the function it is used with
    
        """
        nowTime = datetime.datetime.now()
        keysToDelete = []

        #Traversing on the files dictionary
        for key in uuid_file_dict:
            fileTime = uuid_file_last_modified_datetime_dict[key]
            timediff = nowTime-fileTime

            #Storing the key of the file to be deleted
            if(timediff.seconds > TIMEOUT_TIME and key not in sample_file__list):
                keysToDelete.append(key)

        #Deleting the keys from the dictionary
        for key in keysToDelete:
            uuid_file_dict.pop(key)
            uuid_file_last_modified_datetime_dict.pop(key)
        return func(*args, **kwargs)
    return inner1

# Home Function
@delete_unused_files_decorator
def home(request):
    """
        Returns the Home Page of the application along with the sample file information
    
        Parameters:
        request (Request): Request object which contains all the information about the incoming request
    
        Returns:
        HttpResponse: Http Response of the home file along with a sample file list and images
    
    """
    param = {}
    param["sample_dataset_list"] = sample_file__list
    param["sample_files_image_path_dict"] = sample_files_image_path_dict
    return render(request, 'index.html',param)


# Upload CSV File Function
@delete_unused_files_decorator
def upload(request):

    """
        Upload the file in the incoming request

        Uploads the file in the incoming request, stores it against a random UUID key in the dictionary and also 
        updates the last modified time of the file as the current time and then redirects to /<uuid_key>
    
        Parameters:
        request (Request): Request object which contains all the information about the incoming request
    
        Returns:
        HttpResponse: Http Response of the data view page 
    
    """

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
        return redirect("/")


#Function to load the show data and plot page
@delete_unused_files_decorator
def show_data(request, csv_uuid):

    """
        Returns the Data View Page of the application with the list of column names of the selected data and the 
        graph types supported by the application. If the request method is get then returns the graph of the mentioned 
        type and columns.
    
        Parameters:
        request (Request): Request object which contains all the information about the incoming request
        csv_uuid (string) : Unique UUID Key to access the upload file on the server
    
        Returns:
        HttpResponse: Http Response of the View Data page along with the plot (if the requets contains the information about the
        graph type)
    
    """

    #Loading data into pandas dataframe
    data = load_data(csv_uuid)
    if data is not None:
        
        #Updating the paramater dictionary with initial values
        param = {"data_columns": list(data.columns), "data_values": list(
            data.head(7).values), "graph_list": graph_list}
        param["plot"] = None
        param["isGraphPlotted"] = False
        param["description"] = "Choose A Graph Type, X axis and Y axis to plot the graph of your choice.After plotting the graph you can also train a linear regression or logistic regression model on your data"

        try:
            if (request.method == 'GET'):
                graph_type = request.GET.get("graph")
                y_column = request.GET.get("ycolumn")
                x_column = request.GET.get("xcolumn")
                param["plot"] = None

                #style variable to update the style of the plot
                style = "seaborn-v0_8-darkgrid"

                if (graph_type is not None):

                    #Plotting the graph with the given attributes
                    plot = get_graph(data, graph_type, x_column, y_column, style)
                    param["plot"] = plot
                    param["description"] = graph_descrition[graph_type]
                    param["isGraphPlotted"] = True

            param['csv_uuid'] = csv_uuid
            return render(request, 'show_data.html', param)

        except:
            
            #Updating the paramaters with error values
            param["plot"] = None
            param["description"] = "Some Error Occured!! It seems that your data is not supported by the graph type you have selected"
            param["isGraphPlotted"] = False
            param["Error"] = True

            return render(request,'show_data.html',param)

    else:
        return redirect("/")


#Function to train Linear Regression model
@delete_unused_files_decorator
def linear_regression(request,csv_uuid):

    """
        Returns the Linear Regression Page of the Application along with the data columns list and if there is a GET request then 
        trains the model and returns the model along the training and test RMSE
    
        Parameters:
        request (Request): Request object which contains all the information about the incoming request
        csv_uuid (string): Unique UUID Key to access the upload file on the server
    
        Returns:
        HttpResponse: Http Response of the Linear Regression Page along with the model and training and test RMSE, if attribute 
        selected
    
    """

    #Loading the dataset from stored dictionary
    data = load_data(csv_uuid)
    if data is not None:
        param = {}
        param["column_list"] = list(data.columns)

        if request.method=="GET":
            target_column = request.GET.get("target_column")
            train_test_split = request.GET.get("train_test_split")
            if(target_column is not None):

                #Training the Linear Regression model with the given attibutes, return model = None if data is not compatible
                model,training_rmse,test_rmse = get_trained_linear_regression_model(data,target_column,train_test_split)
                if(model is None):
                    param["error"] = True
                    param["no_error"] = False
                else:

                    #Updating parameters with error values
                    param["model"] = model
                    param["training_rmse"] = training_rmse
                    param["test_rmse"] = test_rmse
                    param["error"] = False
                    param["no_error"] = True

        return render(request, 'linear_regression.html',param)

    else:
        return redirect("/")

#Function to train logistic Regreesion model
@delete_unused_files_decorator
def logistic_regression(request,csv_uuid):

    """
        Returns the Logistic Regression Page of the Application along with the data columns list and if there is a GET request then 
        trains the model and returns the model along the training and test Accuracy
    
        Parameters:
        request (Request): Request object which contains all the information about the incoming request
        csv_uuid (string): Unique UUID Key to access the upload file on the server
    
        Returns:
        HttpResponse: Http Response of the Logistic Regression Page along with the model and training and test Accuracy, if attribute 
        selected
    
    """

    #Loading the dataset stored on the server
    data = load_data(csv_uuid)
    if data is not None:
        param = {}
        param["column_list"] = list(data.columns)

        if request.method=="GET":
            target_column = request.GET.get("target_column")
            train_test_split = request.GET.get("train_test_split")
            if(target_column is not None):

                #Training the logistic regression model with the given attributes, returns None if data is incompatible
                model,training_accuracy,test_accuracy = get_trained_logistic_regression_model(data,target_column,train_test_split)
                if(model is None):
                    param["error"] = True
                    param["no_error"] = False
                else:
                    #Updating the parameters with error values
                    param["model"] = model
                    param["training_accuracy"] = training_accuracy
                    param["test_accuracy"] = test_accuracy  
                    param["error"] = False
                    param["no_error"] = True

        return render(request, 'logistic_regression.html',param)
    else:
        return redirect("/")



# Utility Functions 

#Function to load data from stored dictionary
def load_data(uuid_key):
    """
        Loads the data into a pandas Dataframe, stored as Bytes IO object aagainst the given key
    
        Parameters:
        uuid_key (string) : Unique UUID key to identify the files stored on the server 
    
        Returns:
        pandas.Dataframe : returns the pandas Dataframe of the data stored against the provided key
    
    """

    if uuid_key in uuid_file_dict:
        csv_file = uuid_file_dict[uuid_key]
        uuid_file_last_modified_datetime_dict[uuid_key] = datetime.datetime.now()

        #Seek to point to beginning of the file
        csv_file.seek(0)
        #Reading file into dataframe
        data = pd.read_csv(csv_file)
    else:
        data = None

    return data

def get_graph(data, graph_type, x_column, y_column, style):

    """
        Returs the plotted graph for the given attributes
    
        Parameters:
        data (pandas.Dataframe) : Pandas Dataframe of the data for which the graph has to be plotted
        graph_type (string) : Type of graph to be plotted
        y_column (string) : Name of the column for y-axis
        x_column (string) : Name of the column for x-axis
    
        Returns:
        
        string : utf-8 encoded string of the plotted graph
    
    """

    plot = None
    if graph_type in graph_function_dict:
        plot = graph_function_dict[graph_type](
            data=data, x_column=x_column, y_column=y_column, style=style)
    
    return plot

def get_trained_linear_regression_model(data,target_column,train_test_ratio):

    """
        Returs the trained linear regression model, training rmse, test rmse for the given attributes, returns None for all of them 
        if the data is incompatible
    
        Parameters:
        data (pandas.Dataframe) : Pandas Dataframe of the data for which the model has to be trained
        target_column (string) : Name of the column for the target variable
        train_test_ratio (string) : A numeric string representing the ratio of training data 
    
        Returns:
        
        (string,float,float) : utf-8 encoded string of the trained model, training rmse and the test rmse
    
    """

    #First Checking if all the columns are numeric ( Currently Only supporting Numeric Values )
    try:

        #Converting the data to float
        data = data.astype(float)
    
        #Seperating target and feature attributes
        y = data[target_column]
        X = data.drop(target_column,axis = 1)

        X = np.asarray(X)
        y = np.asarray(y)

        #Splitting the data into training and test sets
        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42,train_size = float(train_test_ratio))

        #Scaling the data with Standard Scaler
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        #Training the Linear Regression model 
        model = LinearRegression()

        model.fit(X_train,y_train)

        #Predicting the values for training and test data 
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        #Calculating the RMSE values for the training and test data sets
        train_rmse = rmse(y_train_pred,y_train)
        test_rmse = rmse(y_test_pred,y_test)

        #Converting the model to utf-8 encoded string
        model_buffer = BytesIO()
        pickle.dump(model,model_buffer)

        model_buffer.seek(0)

        model_value = model_buffer.getvalue()
        model_buffer.close()

        model_base64 = b64encode(model_value)
        model_utf8 = model_base64.decode('utf-8')

        return (model_utf8,train_rmse,test_rmse)

    except:
        return (None,None,None)
        

def rmse(y_pred,y_actual):

    """
        Returs the Root Mean Square Error 
    
        Parameters:
        y_pred (np.array) : Numpy Array of predicted Values
        y_actual (np.array) : Numpy Array of actual Values
    
        Returns:
        
        float : RMSE loss
    
    """

    mse = 1/y_actual.shape[0]*np.sum((y_pred-y_actual)**2)
    rmse = np.sqrt(mse)
    return rmse

def get_trained_logistic_regression_model(data,target_column,train_test_ratio):

    """
        Returs the trained logistic regression model, training accuracy, test accuracy for the given attributes, returns None for all of them 
        if the data is incompatible
    
        Parameters:
        data (pandas.Dataframe) : Pandas Dataframe of the data for which the model has to be trained
        target_column (string) : Name of the column for the target variable
        train_test_ratio (string) : A numeric string representing the ratio of training data 
    
        Returns:
        
        (string,float,float) : utf-8 encoded string of the trained model, training accuracy and test accuracy
    
    """

    #First Checking if all the columns are numeric ( Currently Only supporting Numeric Values )
    try:
        data = data.astype(float)
        
        #Seperating target and feature attributes
        y = data[target_column]
        X = data.drop(target_column,axis = 1)

        X = np.asarray(X)
        y = np.asarray(y)

        #Splitting the data into training and test sets 
        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42,train_size = float(train_test_ratio))

        #Scaling the features using standard scaling
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        #Training the logistic Regression model
        model = LogisticRegression()

        model.fit(X_train,y_train)

        #Precting the Training and Test Values and calculating the accuracy scores
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy(y_train_pred,y_train)
        test_accuracy = accuracy(y_test_pred,y_test)

        #Converting the model to utf-8 encoding 
        model_buffer = BytesIO()
        pickle.dump(model,model_buffer)

        model_buffer.seek(0)

        model_value = model_buffer.getvalue()
        model_buffer.close()

        model_base64 = b64encode(model_value)
        model_utf8 = model_base64.decode('utf-8')

        return (model_utf8,train_accuracy,test_accuracy)
    except:
        return (None,None,None)


def accuracy(y_pred,y_actual):

    """
        Returs the Accuracy 
    
        Parameters:
        y_pred (np.array) : Numpy Array of predicted Values
        y_actual (np.array) : Numpy Array of actual Values
    
        Returns:
        
        float : Accuracy
    
    """

    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_actual[i]:
            count+=1
    
    accuracy = count/len(y_pred)
    return accuracy
    