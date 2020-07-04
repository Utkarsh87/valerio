# ValerioTrials


### Dependencies ###
**Python 3.6.8**<br>

| Python Package | Version |
| ------- | ------- |
| smopy | 0.0.6 |
| networkx | 2.4 |
| pandas | 1.0.3 |
| numpy | 1.18.1 |
| matplotlib | 3.2.0 |
| gdal | 2.4.1(2040100) |

It is advised to freeze these libraries at the above mentioned versions to avoid running into issues related to version incompatibilities as few of the libraries used in this project are not very well maintained.<br>

To install the dependencies(all except GDAL):
```console
foo@bar:~/ValerioTrials$ pip install -r requirements.txt
```

To install GDAL, use the whl file included as instructed below:
```console
foo@bar:~/ValerioTrials$ python -m pip install "path to folder/GDAL-2.4.1-cp36-cp36m-win_amd64.whl"
```
(If python and pip don't work, try the same commands with python3 and pip3)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Run the script ###
The script housing the main algorithm is **Plotting/route plotter.py**<br>
A notebook version of the same is **Plotting/route plotter.ipynb**<br>
After cloning the repo :<br>
**NOTE:** Make sure you are in the **Plotting** folder.<br>

```console
foo@bar:~/ValerioTrials$ cd Plotting
foo@bar:~/ValerioTrials/Plotting$ python "route plotter.py"
```
(If the above command doesn't work, try the same with python3(worked for me :)))

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Sample results ###
* **Source**: Chanakyapuri, Delhi<br>
   **Destination**: Mayur Vihar, Delhi<br>
   **Battery level**: 10%<br>
   **Vehicle**: Revolt RV400<br>
   ![example 1](https://github.com/Utkarsh87/ValerioTrials/blob/master/Plotting/images/10%25_7.png)
   
* **Source**: Sarita Vihar, Delhi<br>
   **Destination**: Rajouri Garden, Delhi<br>
   **Battery level**: 35%<br>
   **Vehicle**: Mahindra E2O P2 Automatic<br>
   ![example 2](https://github.com/Utkarsh87/ValerioTrials/blob/master/Plotting/images/SV-RG%2035%25_5.png) 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Visualise the road network ###
[Source](https://mapshaper.org/) for visualising the shapefile, add the entire zipped folder(Plotting/delhi_highway.zip)<br>

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Inaccuracies and shortcomings ###

* The shapefile is very sparse, a denser shapefile would mean more nodes in the graph and hence a more accurate representation of a given location on the map. Also a denser shapefile would mean including the smaller 'gully' roads which users could take to drastically cut down travel time.
* A better geocoding service(GoogleMaps for instance) would increase the accuracy as remote locations could be fed as locations to the script.
* Use a better tool for plotting the retrieved map(again, GoogleMaps for instance) would lend much greater control over the zoom levels and the POV.
* Charging station database has too few entries(which also perhaps reflects the ground reality of the EV scene in India currently)
* Currently the data and the script both are focussed in the Delhi-NCR region but porting the same functionality for a Pan-India script should be very easy.
* The range calculation script in use is far from representing a real-world scenario.
