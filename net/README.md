# what is "ALEN"?

 **ALEN** is end-to-end network based on attention mechanism for processing low-light images

# there are 2 type's "Attention"

* spatial => focus on denoising by taking advantage of the non-local correlation in the image => **remove noise**

![mixed_attention_block](https://user-images.githubusercontent.com/57025585/96367100-cac30380-1186-11eb-8ba8-fdbeec157e6f.png)

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

# network Architecture

![architecture](https://user-images.githubusercontent.com/57025585/96368949-e253b980-1191-11eb-8c2b-2787d56de835.png)

network is in the form of **U-net**.it has 2 parts, encoder and decoder.

* encoder => employ several mixed attention blocks and ISLs to obtain semantic features.
* decoder => employ multiple conv layers and transposed conv to restore high-resolution features from semantic features.

 finally,the estimated image(**Ie**) is obtained after a pixel shuffle operation from a 12-channel feature map.

## term of network Architecture

* ISL => Inverted Shuffle Layer
* MAB => Mixed Attention Block
* CAB => Channel Attention Block

## Inverted Shuffle Layer

pooling operation usually abandons useful information in the forword process whether it is max pooling or average pooling.

-> ISL ( new proposed pooling operation) includes inverted shuffle and convolution operation

-> after an inverted shuffle operation, the size of the feature map reduces to half of the original and the number of channels quadruples.

-> convolution layer with 1x1 kernels is perfoemed after the inverted shuffle,which plays a role in selecting useful information while compressing the number of channels.

-> ISL not only has the effect of reducing the computation as a pooling layer but also makes the network more flexible to select features.
