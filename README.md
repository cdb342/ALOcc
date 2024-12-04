<div id="top" align="center">

    
### **ALOcc:** *Adaptive Lifting-based 3D Semantic Occupancy and Cost Volume-based Flow Prediction*


[![arXiv](https://img.shields.io/badge/arXiv-2411.07725-b31b1b.svg)](https://arxiv.org/abs/2411.07725)
[![License: Apache2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](#licenseandcitation)

</div>


ALOcc is a powerful vision-based 3D semantic occupancy and flow prediction tool, primarily designed for applications like autonomous driving. The project focuses on improving 3D occupancy prediction and flow estimation.

<p align="center">
  <img src="assets/1.png" width="38%" style="margin-right: 10px;">
  <img src="assets/2.png" width="48%">
</p>


## Main Results

<table>
<caption><b>3D semantic occupancy prediction performance on Occ3D (training w/ camera visible mask).</b></caption>
  <thead>
    <tr>
      <th align="left">Model</th>
      <th align="center">Backbone</th>
      <th align="center">Input Size</th>
      <th align="center">mIoU<sub>D</sub><sup>m</sup></th>
      <th align="center">mIoU<sup>m</sup></th>
      <th align="center">FPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>ALOcc-2D-mini</b></td>
      <td align="center">ResNet-50</td>
      <td align="center">256 × 704</td>
      <td align="center">35.4</td>
      <td align="center">41.4</td>
      <td align="center">30.5</td>
    </tr>
    <tr>
      <td><b>ALOcc-2D</b></td>
      <td align="center">ResNet-50</td>
      <td align="center">256 × 704</td>
      <td align="center">38.7</td>
      <td align="center">44.8</td>
      <td align="center">8.2</td>
    </tr>
    <tr>
      <td><b>ALOcc-3D</b></td>
      <td align="center">ResNet-50</td>
      <td align="center">256 × 704</td>
      <td align="center">39.3</td>
      <td align="center">45.5</td>
      <td align="center">6.0</td>
    </tr>
  </tbody>
</table>


<table>
  <caption><b>3D semantic occupancy prediction performance on Occ3D (training w/o camera visible mask).</b></caption>
  <thead>
    <tr>
      <th align="left">Method</th>
      <th align="center">Backbone</th>
      <th align="center">Input Size</th>
      <th align="center">mIoU</th>
      <th align="center">RayIoU</th>
      <th align="center">RayIoU<sub>1m, 2m, 4m</sub></th>
      <th align="center">FPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>ALOcc-2D-mini</b></td>
      <td align="center">ResNet-50</td>
      <td align="center">256 × 704</td>
      <td align="center">33.4</td>
      <td align="center">39.3</td>
      <td align="center">32.9, 40.1, 44.8</td>
      <td align="center">30.5</td>
    </tr>
    <tr>
      <td><b>ALOcc-2D</b></td>
      <td align="center">ResNet-50</td>
      <td align="center">256 × 704</td>
      <td align="center">37.4</td>
      <td align="center">43.0</td>
      <td align="center">37.1, 43.8, 48.2</td>
      <td align="center">8.2</td>
    </tr>
    <tr>
      <td><b>ALOcc-3D</b></td>
      <td align="center">ResNet-50</td>
      <td align="center">256 × 704</td>
      <td align="center">38.0</td>
      <td align="center">43.7</td>
      <td align="center">37.8, 44.7, 48.8</td>
      <td align="center">6.0</td>
    </tr>
  </tbody>
</table>


<table>
  <caption><b>3D semantic occupancy and flow prediction performance on OpenOcc.</b></caption>
  <thead>
    <tr>
      <th align="left">Method</th>
      <th align="center">Backbone</th>
      <th align="center">Input Size</th>
      <th align="center">Occ Score</th>
      <th align="center">mAVE</th>
      <th align="center">mAVE<sub>TP</sub></th>
      <th align="center">RayIoU</th>
      <th align="center">RayIoU<sub>1m, 2m, 4m</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>ALOcc-Flow-2D</b></td>
      <td align="center">ResNet-50</td>
      <td align="center">256 × 704</td>
      <td align="center">42.1</td>
      <td align="center">0.537</td>
      <td align="center">0.427</td>
      <td align="center">40.5</td>
      <td align="center">34.3, 41.3, 45.8</td>
    </tr>
    <tr>
      <td><b>ALOcc-Flow-3D</b></td>
      <td align="center">ResNet-50</td>
      <td align="center">256 × 704</td>
      <td align="center">43.0</td>
      <td align="center">0.556</td>
      <td align="center">0.481</td>
      <td align="center">41.9</td>
      <td align="center">35.6, 42.8, 47.4</td>
    </tr>
  </tbody>
</table>


## Bibtex
Please consider citing our paper if it is helpful for your research:
```BibTeX
@article{chen2024alocc,
  title={ALOcc: Adaptive Lifting-based 3D Semantic Occupancy and Cost Volume-based Flow Prediction},
  author={Chen, Dubing and Fang, Jin and Han, Wencheng and Cheng, Xinjing and Yin, Junbo and Xu, Chenzhong and Khan, Fahad Shahbaz and Shen, Jianbing},
  journal={arXiv preprint arXiv:2411.07725},
  year={2024}
}
```