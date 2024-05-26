# Language Modeling Report

In this report, we present the implementation and evaluation of character-level language models using vanilla RNN and LSTM architectures. We adhere to the requirements outlined for the project.

## 1. Dataset Pipeline (`dataset.py`)

We implemented a custom dataset pipeline in `dataset.py` to provide data to our models. This pipeline loads and preprocesses the Shakespeare dataset. Specifically, it constructs a character dictionary and creates sequences of length 30, splitting the data into chunks with appropriate target sequences.

## 2. Model Implementation (`model.py`)

We implemented both vanilla RNN and LSTM models in `model.py`. For both architectures, we stacked multiple layers (2 in main.py) to explore if it improves performance. The implementation follows the instructions provided in the file as comments.

We set the following hyper parameters as follows

batch_size = 64
seq_length = 30
hidden_size = 128
num_layers = 2
lr = 0.002
epochs = 30

## 3. Training Script (`main.py`)

The `main.py` script is responsible for training our models. During training, we monitor the training process using average loss values of both training and validation datasets. We utilized torch's CrossEntropyLoss as our cost function and employed the Adam optimizer.

## 4. Evaluation and Reporting (report)

We plotted the average loss values for both training and validation datasets to evaluate the performance of our models. Additionally, we compared the language generation performances of vanilla RNN and LSTM models in terms of loss values for the validation dataset. This comparison helps us assess which model architecture performs better for this task.

As the result of the experiment, the LSTM outperforms the RNN. The following graph depicts the lapse of each model at every epoch.
![Alt text](./images/model_results.png)
Traning loss:
RNN starts with less loss over LSTM. However, LSTM's loss reduces near the epoch 7 (see the graph line blue(RNN) and LSTM(Green))
They both keep decreasing as further epochs are iterated

Validation loss:
RNN starts with less loss over LSTM. However, LSTM's loss reduces near the epoch 6 (see the graph line Orange(RNN) and LSTM(Red))
The RNN loss converges around the loss 1.6 unlike the that of LSTM which slightly increases after the epoch 19


## 5. Character Generation (`generate.py`)

We implemented character generation functionality in `generate.py` using the trained model with the best validation performance. We selected the model that exhibited the lowest validation loss. The script generates at least 100 characters of 5 different samples from different seed characters.

## 6. Softmax Function with Temperature Parameter (Report)

The softmax function with a temperature parameter T can be written as:

$$
y_i = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}
$$

<br>

We experimented with different temperatures when generating characters and observed their impact on the generated results. In our analysis, we discuss how varying the temperature affects the plausibility and diversity of the generated text. We chose the LSTM model that outperforms the vanilla RNN.

<br>

The samples are the Shakespeare lines that we experimented for kindly see the results of each sample with different temperatures 0.2, 0.5, 0.8, 1.0, 1.2
<br><br>
Sample1: "He cannot temperately transport his honours",
<br>
Sample2: "Were he to stand for consul, never would he",
<br>
Sample3: "We have power in ourselves to do it,",
<br>
Sample4: "Good night, good night! parting is such sweet sorrow,",
<br>
Sample5: "And with thy scorns drew'st rivers from his eyes"

<br>

## Analysis
The low-temperature hyperparameter keeps with the general sentence structures in the provided corpus, which makes generated texts too repetitive
<br>
On the other hand, the sentence structures are broken by using higher temperatures. 
It may cancel out the influence of the predicted result (z_i) and cause a model to explore the creative structures and words that were not trained
<br>
These are some results at different temperatures to check the plausibility of generated texts.
we concluded that using 0.5 and 0.8 is more plausible than other settings in terms of the grammar structure (using 0.2 is too repetitive)


## generated_text temperature  0.2
### Sample 1
**original characters:**  
He cannot temperately transport his honours  

**generated_text:**  
to the people,
That I must not the common the people and the people and the people and the many tha

---

### Sample 2
**original characters:**  
Were he to stand for consul, never would he  

**generated_text:**  
did stop the people  
That I have not this in the common mercy the people and the people.  

CORIOLANUS

---



### Sample 3
**original characters:**  
We have power in ourselves to do it,  

**generated_text:**  
and the way,  
And the people and the common me to the wars and did not a grace with him.  

CORIOLANUS

---

### Sample 4
**original characters:**  
Good night, good night! parting is such sweet sorrow,  

**generated_text:**  
should be so shall the people my heart the deed the people stribent of the deeds the many the commo

---

### Sample 5
**original characters:**  
And with thy scorns drew'st rivers from his eyes  

**generated_text:**  
,  
Which we have not the common me to the people and the princely s

## generated_text temperature  0.5
### Sample 1
**original characters:**  
He cannot temperately transport his honours  

**generated_text:**  
Of the world the people and struck with the feed and his healls and private the matter, when
That w

---
### Sample 2
**original characters:**  
Were he to stand for consul, never would he  

**generated_text:**  
will shall thee a great of them.  

**MENENIUS:**  
There's love, he is men a friends, if you devil the sea

---

### Sample 3
**original characters:**  
We have power in ourselves to do it,  

**generated_text:**  
And have pardon me as the Romans the city of yours,  
He haspitions the world they devil the market-p

---

### Sample 4
**original characters:**  
Good night, good night! parting is such sweet sorrow,  

generated_text:
and he's men to the duke age her he wounds should streck the state of my saint with ease  
and the ma

---

### Sample 5
**original characters:**  
And with thy scorns drew'st rivers from his eyes  

**generated_text:**  
, which, by still thee, and the market-place.  

MENENIUS:
No, being strong of the city the body had I

## generated_text temperature  0.8
### Sample 1
**original characters:**  
He cannot temperately transport his honours  

**generated_text:**  
   of so.

CORIOLANUS:
Nor sweat this one back; to the great she thrust now now, therefore your grace,

---

### Sample 2
**original characters:**  
Were he to stand for consul, never would he  

**generated_text:**  
will accemset me and the did agre; the weary! but we will not your days, and disposition.  

GLOUCEST

---

### Sample 3
**original characters:**  
We have power in ourselves to do it,  

**generated_text:**  
my stirit,  
We have noble your Calain I dull go the remember  
For with this hands, the king 'Gore so,

---

### Sample 4
**original characters:**  
Good night, good night! parting is such sweet sorrow,  

**generated_text:**  
God gothing of our day, with mine arms  
Ingrace of my commands not for my both and forget them recom

---

### Sample 5
**original characters:**  
And with thy scorns drew'st rivers from his eyes  

**generated_text:**  
,  
To-day a man to arm, come along but the wife,  
To the supple him a cerricled many a kingham?  

Secon

## generated_text temperature  1.0
### Sample 1
**original characters:**  
He cannot temperately transport his honours  

**generated_text:**  
   diend many good full taknessed, my how oundage
I moreable force, thou that do men, yours, she neck 

---

### Sample 2
**original characters:**  
Were he to stand for consul, never would he  

**generated_text:**  
;  
The honour in his more priteen  
For aught than deeds and my son hou before for my lord; grand's die

---

### Sample 3
**original characters:**  
We have power in ourselves to do it,  

**generated_text:**  
that I be lengmens, in alive uncles.  

MENENIUS:
On them a knidest it.  

GLOUCESTER:
Then: and yet fa

---

### Sample 4
**original characters:**  
Good night, good night! parting is such sweet sorrow,  

**generated_text:**  
struck before eye  
To puny you, sir!  
Rast hanged I have me.  

CORIOLANUS:
You are men with a limb.  

L

---

### Sample 5
**original characters:**  
And with thy scorns drew'st rivers from his eyes  

**generated_text:**  
, whom  
Like he senatons, who are puty.  

VOLUMNIA: 
Spous that thence to a friends, it a mainant them

## generated_text temperature  1.2
### Sample 1
**original characters:**  
He cannot temperately transport his honours  

**generated_text:**  
  ,
And, scarces well, yellow mayier the city from it
moltinus, my masseet
are hroat on all the Edward

---

### Sample 2
**original characters:**  
Were he to stand for consul, never would he  

**generated_text:**  
: amed, well view 'judgmed, not uspe his neplieding follow  
make in theecherefore, there behel;  
there

---

### Sample 3
**original characters:**  
We have power in ourselves to do it,  

**generated_text:**  
I malked to my diseaint thy, myself you. En Tife of 'en and stip knoward his pass.  

CORIOLANUS:
Rai

---

### Sample 4
**original characters:**  
Good night, good night! parting is such sweet sorrow,  

**generated_text:**  
little our veins to life done, let us the ictore.  

GLOUCESTER:
By did his spent, you shall flidius,

---

### Sample 5
**original characters:**  
And with thy scorns drew'st rivers from his eyes  

**generated_text:**  
,  
Who die be my stervet the way-ingly enouged, go love,  
Bithon us i' the pack'd me; and again fault,

## generated_text at temperature  1.5
### Sample 1
**original characters:**  
He cannot temperately transport his honours  

**generated_text:**  
They we coupsuace in the spidedio't,
Est wife
A spary; and I muse nay; we'lcker lip night,
Whose na

---

### Sample 2
**original characters:**  
Were he to stand for consul, never would he  

**generated_text:**  
, repore! Is that, my heath,  
To-doton appeslish Deny yourself:  
Readided in I will, fnobed off,--Prec

---

### Sample 3
**original characters:**  
We have power in ourselves to do it,  

**generated_text:**  
do by this chaon forget  
Stiraching them as madgured  
it were you meefure; he make, Ine. I had  
fillse

---

### Sample 4
**original characters:**  
Good night, good night! parting is such sweet sorrow,  

**generated_text:**  
-hight! So, this,  
efpess brive onculaterdly achely pople tre--tyrmys, from himpits corrow;  
Yes, go,

---

### Sample 5
**original characters:**  
And with thy scorns drew'st rivers from his eyes  

**generated_text:**  
,  
More Rome, was died whice:-what Warquigns: princest  
As fearoof giveges-sate. Your, Lidius:  
Come, b
