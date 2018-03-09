# mvcnn-spherepano

This project is following Hang Su's mvcnn code at https://github.com/suhangpro/mvcnn, where the dependices and usage can all be referred. The project is developed on MATLAB 2015a, containing works of network training, retrieval testing and classification testing, while the data for training should be mannually prepared in advance.

For training, the initial input network 'net-fit.mat' is modified from mvcnn's 'imagenet-matconvnet-vgg-m.mat', with filters in one hidden layer are clipped by half, to fit the input spherical projected images of size 512*256.

If you use any part of the code from this project, please cite:

  @article{Feng17spherepano,
  author    = {Yuanli Feng and Meng Xia and Penglei Ji and Xiao Zhou and Ming Zeng and Xinguo Liu},
  title     = {Deep Spherical Panoramic Representation for 3D Shape Recognition},
  booktitle = {Journal of Computer-Aided Design & Computer Graphics}, 
  year      = {2017}}
