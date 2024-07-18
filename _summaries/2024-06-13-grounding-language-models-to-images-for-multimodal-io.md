---
layout: summary
title: "Grounding Language Models to Images for Multimodal Inputs and Outputs"
giscus_comments: true
bib_id: 2301.13823v4
---

### Motivation

In my investigations for Multimodal RAG, I realized that whatever a language transformer and visual transformer attend to should be the same. If they are vectors, they should be similar. This is true, but only if the modalities are the same. This implies that a sentence and image should be similar IFF they are in the same space. Some more research told me this is called "Grounding". This paper discusses this, and I'll provide a 3 sentence review after revising it.

### Paper Abstract

We propose an efficient method to ground pretrained text-only language models to the visual domain, enabling them to process arbitrarily interleaved image-and-text data, and generate text interleaved with retrieved images. Our method leverages the abilities of language models learnt from large scale text-only pretraining, such as in-context learning and free-form text generation. We keep the language model frozen, and finetune input and output linear layers to enable cross-modality interactions. This allows our model to process arbitrarily interleaved image-and-text inputs, and generate free-form text interleaved with retrieved images. We achieve strong zero-shot performance on grounded tasks such as contextual image retrieval and multimodal dialogue, and showcase compelling interactive abilities. Our approach works with any off-the-shelf language model and paves the way towards an effective, general solution for leveraging pretrained language models in visually grounded settings.

### Three Important Points

#### 1. The Need for Visual Inputs

The authors note that current Large Language Models are grounded in the language modality, since they are trained on large corpuses of internet text data. However, this means that LLMs are unable to process multimodal data, such as audio or visual data, and are inherently limited in that respect. The authors note that visual data can be information rich and serve to ground LLMs in the real world and make them more effective than just being trained on internet text data. This paper focuses on the incorporation and interleaving of visual data into an pretrained LLM, with a focus on Image Captioning and Image Retrieval given a prompt. This approach is efficient and relevant since their architecture leverages existing visual models and LLM models by learning small weight matrices and embeddings for only a single new token, meaning that instead of training a large costly model, training is relatively fast and efficient.

#### 2. FROMAGe Model

This paper introduces the FROMAGe model. The FROMAGe model works by taking an existing LLM and Visual Encoder, freezing their parameters, and learning transformations between image and text embedding spaces to perform image $\rightarrow$ text and text $\rightarrow$ image transformations. FROMAGe was designed for image captioning and image retrieval, and does these steps separately. It should be noted that this is possible since a previous paper, LIMBeR, found that learnt representations of vision and language models are equivalent up to a linear transformation, which can be learned as a linear layer.

{% include figure.liquid
    path="/assets/img/summaries/image_captioning.jpg"
    width="600px"
    class="z-depth-1"
%}

For Image Captioning, FROMAGe learns a single linear layer called $W_C$ that serves to translate the output of the visual encoder into a token that is then prepended to the caption text corresponding to the image. This lines up with previous work on image captioning, which generates text tokens conditioned on a visual prefix. The standard log likelihood loss is used to learn the linear layer.

{% include figure.liquid
    path="/assets/img/summaries/image_text_retrieval.jpg"
    width="600px"
    class="z-depth-1"
%}

For image retrieval, FROMAGe learns 2 separate linear layers, called $W_t, W_i$ for translating text $\rightarrow$to image and image $\rightarrow$ text respectively. The authors used the InfoNCE loss and Contrastive Learning to learn these linear layers. Notably, the authors also learned a new `[RET]` token and the associated embeddings. This allows FROMAGe to be able to produce a stronger text representation for the image as well as be able to generate the new `[RET]` token at inference time to allow for the interleaving of images and text.

#### 3. FROMAGe vs CLIP

The authors perform various experiments comparing FROMAGe to other models, most notably CLIP. CLIP functions by learning a correspondence between which texts and images were paired together during training time. CLIP is a powerful model, but an admittedly rigid and inflexible one, since instead of learning translation layers like FROMAGe does, CLIP learns pairings of data and solely relies on in context training to function. The experiments support this intuition, as they show that given more context than just 1 caption, FROMAGe outperforms CLIP and other related models at recall at K.

Recall at K is a metric that measures, given K recommendations from the model, the proportion of relevant results in these top K recommendations compared to the number of relevant results in the dataset.

Further experiments show that FROMAGe performs well in visual dialogue tasks and in general, effectively leverages the visual data that it has access to that other LLM or mixed-modality models do not effectively use. FROMAGe, when provided with both more visual and text input, increases its recall at 1 (R @ 1) measure, which indicates that it does better at finding the most relevant answer from the given dataset (Visual Storytelling dataset).

{% include figure.liquid
    path="/assets/img/summaries/model_exp_1.jpg"
    width="600px"
    class="z-depth-1"
%}

{% include figure.liquid
    path="/assets/img/summaries/model_exp_2.jpg"
    width="600px"
    class="z-depth-1"
%}

### Most Glaring Deficiency

The authors did not show the effect of just providing more images to FROMAGe to understand empirically how much more information a single image provides to the model, and if there are diminishing returns on providing more images or not. In the above results, we notice that the authors provide more captions and images at the same time, which makes it difficult to tell if text and image data need to strictly be coupled to provide better performance, or if just providing more image data can improve the R @ k metric.

### Conclusions for Future Work

FROMAGe introduces some of the machineries and frameworks for thinking about and leveraging more general multimodal models. The notion of multimodality leveraging translations between pre-trained representations instead of explicitly training to learn which images and sentences were paired together, like CLIP, motivate more general architectures that can be used for more general use cases in the future.
