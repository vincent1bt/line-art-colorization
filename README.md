
# Line Art Colorization

The code from this repository includes a implementation of the models from the [Deep Line Art Video Colorization with a Few References](https://arxiv.org/abs/2003.10685) paper.

You can read more about this work in [this blog post](
https://vincentblog.link/posts/line-art-colorization-using-a-deep-learning-model).

Here You can find a color network model which work is colorize line art images. To train this network You need 3 types of input images, color images, line art images and distance map images.

You can generate part of the images using the **data_preprocessing.py** script. I recommend you to read the [blog post](
https://vincentblog.link/posts/line-art-colorization-using-a-deep-learning-model) to know more about the data obtention.

You need to save color images/frames from different shots in the **final_shots** folder, the **data_preprocessing.py**  will take these images and generate line art and distance images.

To train the model you need a folder structure like:

```
...
│ 
├── model_blocks.py
├── loss_functions.py
│ 
├── final_shots (Folder) 
│   ├── shot336 (Folder) 
│   │   ├── 678.jpg
│   │
│   └── shot124 (Folder) 
│   └── shot567 (Folder) 
│ 
├── data_preprocessing.py
...
```

Where each folder in the **final_shots** folder contains frames that are similar to each other.

The **data_preprocessing.py** script uses a model called SketchKeras which original source code is [here](https://github.com/lllyasviel/sketchKeras). The SketchKeras model generates line art images from color images. Furthermore, the **ndimage.distance_transform_edt** function from **scipy**  is used to generate distance map images from the line art images. All these images are used to train the color network.

Run the model:

```
python data_preprocessing.py
```

```
python training.py --epochs=350
```

## Test the Model

You can also test the pre-trained model, in this case you can put some frames in the **test_shots** folder and use the following command:

```
python data_preprocessing.py --generate_test_files
```

This will generate line art and distance images for the frames in the **test_shots** folder.

You need the pre-trained model and put it in the **weights** folder, then run:

```
python test_model.py --batch_size=3
```

You need to adjust the batch size using the ```--batch_size=1``` argument.


If you already have line art images, you can generate only the distance map images:

```
python data_preprocessing.py --generate_test_files --generate_distance_map_only
```

The folder structure is similar when we train the model but using the **test_shots** folder:


```
...
│ 
├── training.py
│ 
├── test_shots (Folder) 
│   ├── color
│   │   ├── shot336 (Folder) 
│   │
│   └── line_art_shots (Folder) 
│   └── distance_map_shots (Folder) 
│ 
├── test_model.py
...
```

Take into account that the model was trained using frames from the anime my little witch academia and the input size of the model is **256x256x3**, thus, images look fine at that size but worse if we increase their size also the colors that the model assign are similar to the colors of the anime.

## Temporal Constrain Network and LearnableTSM

In the **temporal_constrain_model.py** you can find the implementation of the Temporal Shift Module that was introduced in the paper [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383). You can also read more about this module and network in [my blog post](
https://vincentblog.link/posts/line-art-colorization-using-a-deep-learning-model)


## Electron App

You can also read about an Electron app that package the model to deploy it as an app [here](
https://vincentblog.link/posts/line-art-colorization-using-a-deep-learning-model) and the source code of the app [here](https://github.com/vincent1bt/Line-art-colorization-electron-app).


