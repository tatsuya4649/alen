# what is "ALEN"?

 **ALEN** is end-to-end network based on attention mechanism for processing low-light images

# there are 2 type's "Attention"

* spatial => focus on denoising by taking advantage of the non-local correlation in the image => **remove noise**

![mixed_attention_block](https://user-imgaes.githubusercontent.com/57025585/96367007-30fb5680-1186-11eb-9d3f-8e42d2af50f0.png)

* channel => guide the network to refine redundant color features => **color restoration**

![channel_attention_block](https://user-images.githubusercontent.com/57025585/96366959-e548ad00-1185-11eb-9f56-15367d19dba5.png)

# network haves new block( mixed attention block)

mixed attention block is to effectively fuse local and global features.

the proposed mixed attention block effectively suppressed undesired chromatic aberration and noise.

# propose new pooling layer

**Inverted Shuffle Layer**(ISL) is to adaptively select important information from feature maps.

this idea from considering that the max pooling layer often brings about information loss.

# network formula

**Ie = F(Ir,theta)**

F => proposed network
theta => parameters of network
Ir => low-light raw image Ir
Ie = estimated color image

## network Architecture

network is in the form of **U-net**.it has 2 parts, encoder and decoder.

* encoder => employ several mixed attention blocks and ISLs to obtain semantic features.
* decoder => employ multiple conv layers and transposed conv to restore high-resolution features from semantic features.

 finally,the estimated image(**Ie**) is obtained after a pixel shuffle operation from a 12-channel feature map.

## term of network Architecture

* ISL => Inverted Shuffle Layer
* MAB => Mixed Attention Block
* CAB => Channel Attention Block

