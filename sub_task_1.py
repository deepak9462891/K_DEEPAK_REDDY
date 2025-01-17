from IPython import get_ipython
get_ipython().run_line_magic('reset', '-f') # Clear any cached modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
# Import necessary layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
!pip install google.colab
from google.colab import drive
drive.mount('/content/
data_dir = "/content/gdrive/MyDrive/Train"
img_size = (256,256)  
batch_size = 

datagen = ImageDataGenerator(
    rescale=1./255,        
    validation_split=0.2 
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)



val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)


from tensorflow.keras import Sequential
from tensorflow.keras import layers

model = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(256,256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense( 256, activation='relu'),
    layers.Dropout(0.5),  
    layers.Dense(1, activation='sigmoid')  
])


model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(
    train_data,
    epochs=2,
    validation_data=val_data,
)

#-------------------------------------------------------------------------------------#


import pandas as pd
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import re

output_csv_path = "/content/gdrive/MyDrive/submission.csv"

 
test_folder = "/content/gdrive/MyDrive/Test_Images"
img_size = (256,256)  

# Get all image file paths from the test folder
test_images = glob(f"{test_folder}/*")


# Store results in lists
image_ids = []

labels = []

# Iterate through all test images
for img_path in test_images:
    # Extract image ID from the path (numerical part)
    image_id = int(re.search(r'\d+', img_path.split('/')[-1]).group())

    
    img = load_img(img_path, target_size=img_size)  #pre processing 
    img_array = img_to_array(img) / 255.0          
    img_array = np.expand_dims(img_array, axis=0) 


    
    prediction = model.predict(img_array)

    
    label = "Real" if prediction[0][0] > 0.5 else "AI"

    # Append = add
    image_ids.append(image_id)
    labels.append(label)

# Sorting image_ids and labels 
sorted_data = sorted(zip(image_ids, labels))
image_ids, labels = zip(*sorted_data)


results = pd.DataFrame({
    "Id": ["image_" + str(id) for id in image_ids],
    "Label": labels,
})


results.to_csv(output_csv_path, index=False)
