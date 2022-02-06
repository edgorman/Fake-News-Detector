# Fake-News-Detector

> The MediaEval 2015 "verifying multimedia use" task aims to test automatic ways to classify viral social media content propagating fake images or presenting real images in a false
context. After a high impact event has taken place, a lot of controversial information goes viral on social media and investigation needs to be carried out to debunk it and decide whether the shared multimedia represents real information.

_Description from the [MediaEval 2015](http://www.multimediaeval.org/mediaeval2015/) webpage_

This is my implementation for problem given above which uses the natural language toolkit and scikit-learn to design and test a machine learning model to identify fake posts on social media. I was able to achieve a final F1 score of 0.92 by using a hard-voting classifier across three models trained on feature subsets.

See ```docs/Report.pdf``` to read my analysis of the task and final conclusions.

## Installation
Use the following command to clone the respository:
```
cd your/repo/directory
git clone https://github.com/edgorman/Fake-News-Detector
```

Create your Python environment using the environment.yaml file:
```
conda env create --file environment.yaml
```

And then activate it use conda:
```
conda activate fake-news-detector
```

## Usage
To train and test the models mentioned in my report:
```
python fake-news-detector/main.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
