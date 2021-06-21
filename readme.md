#Graph-tcn for Micro-Expression Recognition：  
__A Novel Graph-TCN with a Graph Structured Representation for Micro-expression Recognition__[1]

-------------------------------------------------------------------------------------------------------------------
## contents:  
1.__preprocessing__：Using the Learning-Based Video Motion Magnification [2] to magnify the micro-expression. The remaining preprocessing include commonly used alignment, cropping, resizing and extracting 7x7 patches of eyebrows and mouth facial landmarks.  
2.__mytcn.py and mymodel.py__ together form a complete network model.  
3.__test-casme2-5class.py__ can reproduce the recognition result.(73.98% on casme2 of 5 class).  
4.The __data__ folder are utilized in __stage of 3. :__  
&#8195;In __data/mat__ folder, __feature-5000-gai.mat__  is the feature matrix.  
&#8195;In __data/model__ folder, 26 __newbestxx.pth__ files are the pre-trained model of loso.  
&#8195;In __data/test__ folder, 26 testxx.txt are the index and label of test samples, and the index can query the number of rows in __feature-5000-gai__.mat

##Required Package:  
Torch：1.2.0,    
Torchvision：0.4.0,  
Scipy：1.2.1,  
Numpy：1.16.4.  

##Datasets：
Application website:  
CASMEII:http://fu.psych.ac.cn/CASME/casme2.php  
SAMM: http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php  
SMIC: http://www.cse.oulu.fi/SMICDatabase  

## reference
[1] our paper:<https://dl.acm.org/doi/10.1145/3394171.3413714>   
[2] Learning-Based Video Motion Magnification (paper):<https://arxiv.org/abs/1804.02684>  
&#8195;&#8194;Learning-Based Video Motion Magnification (code in TensorFlow):<https://github.com/12dmodel/deep_motion_mag>  
&#8195;&#8194;Learning-Based Video Motion Magnification (code in pytorch):<https://github.com/ZhengPeng7/motion_magnification_learning-based>  
