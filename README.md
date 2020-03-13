# Historical Handwritten Text Recognition
---
Work in progress handwritten text recognition system uniquely suited to handle historical data. Adapted from https://github.com/githubharald/SimpleHTR with the following proposed improvements:
- [x] Update to run on TensorFlow 2
- [ ] Implement word segmenter
- [ ] Add more CNN layers
- [ ] Implement topological preprocessing
- [ ] Implement deslanting
- [ ] Train on historical data
### Implement word segmenter
---
The network was designed to identify individual words. The end goal of this project is a system that can take an image as input and output a text file. To this end, it will be necessary to create a subsytem that breaks the document image into an ordered series of word images.
### Topological Preprocessing
---
Topological data analysis, specifically persistent homology (PH), [has been shown](https://arxiv.org/pdf/1905.12200.pdf) to have many novel applications in machine learning. I intend to use a topological layer to regularize data. PH can be used to minimize the number of local maxima in an image. For the problem of optical character recognition, this has been used to improve the visual quality of a character while also reducing noise. Historic documents often have worse image quality due to such factors as fading ink, crumbling paper, etc. Historic documents are also prone to noise because of the quality of the paper on which they are written.
### Train on Historical Data
---
One challenge facing historical text recognition is the changing style of handwriting over time. The main goal of this project is to create a network that is trained on historical data.
