Currently, the script runs with saved correspondence points for all faces.  If you want to select your own, comment out the getPoints() function and comment the hard coded points line in the same function.

Run python3 main.py and the midway face, face morph images used to make the gif, the mean face of the population, caricatures, and bells and whistles will be saved in the working directory.  For all the other project outputs, they are inside function loops and can be uncommented, then running python3 main.py will also output these.  I left them commented out to make the loops less confusing.

My hardcoded points (so I didn’t have to continuously reelect them) are in a txt file called saved_points.txt.

The gif images are output to a dir called gif.

To run the mean population portion of the project, all the files from the danish data set must be in a directory called data.