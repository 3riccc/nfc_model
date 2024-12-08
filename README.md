# Coarse-graining network flow through statistical physics and machine learning

This repository will contain the PyTorch implementation of:
<br>

**Coarse-graining network flow through statistical physics and machine learning**<br>
Zhang Zhang, Arsham Ghavasieh, Jiang Zhang, Manlio De Domenico<sup>\*</sup><br>

(<sup>\*</sup>: Corresponding author) <br>
[Download PDF](https://www.researchsquare.com/article/rs-3503708/v1)<br>

<p align="center">
  <img src="./architecture.png" width="600px" alt="">
</p>


### Abstract: 

Information dynamics plays a crucial role in complex systems, from cells to societies.  Recent advances in statistical physics have made it possible to capture key network properties, such as flow diversity and signal speed, using entropy and free energy. However, large system sizes pose computational challenges. We use graph neural networks to identify suitable groups of components for coarse-graining a network and achieve a low computational complexity, suitable for practical application. Our approach preserves information flow even under significant compression, as shown through theoretical analysis and experiments on synthetic and empirical networks. We find that the model merges nodes with similar structural properties, suggesting they perform redundant roles in information transmission. This method enables low-complexity compression for extremely large networks, offering a multiscale perspective that preserves information flow in biological, social, and technological networks better than existing methods mostly focused on network structure.

### Requirements

- Python 3.7.0
- Pytorch 2.0.1
- torch_geometric 2.4.0

### To start fast, please see the following tutorials:

- With this tutorial you can easily train a model to coarse grain your network data
[Tutorial1](https://github.com/3riccc/nfc_model/blob/main/tutorial_for_a_fast_start_on_BA_net_renormalization.ipynb)

- With this tutorial you can load the pre-trained model to coarse grain your network data
[Tutorial1](https://github.com/3riccc/nfc_model/blob/main/load_pre_trained_model_and_apply.ipynb)



### Cite
If you use this code in your own work, please cite our paper:
```
Zhang, Z., Ghavasieh, A., Zhang, J., & De Domenico, M. (2023). Network Information Dynamics Renormalization Group.

```

