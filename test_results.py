import cPickle as pickle
import nltk
import numpy as np
from compute_bleu import compute_bleu
import sys
sys.path.insert(0,'/home/shaunak/ProjectSSSP/FULL_S/caption_generator_resnet/nlg_eval_master')
from nlgeval import NLGEval 

def bleu_score(hypotheses, references, weights):
	return nltk.translate.bleu_score.corpus_bleu(references, hypotheses, weights=weights)

captions = pickle.load( open( "predicted_captions.p", "rb" ) )

image_captions_pair = pickle.load( open( "actual_captions.p", "rb" ) )

imgs = []
img_dir = 'Flickr8k_text/Flickr_8k.testImages.txt'
with open(img_dir, 'rb') as f_images:
	imgs = f_images.read().strip().split('\n')


hypotheses=[]
references = []

hypotheses_spice = []
references_spice = []
bleus=[]
for img_name in imgs:
	hypothesis = captions[img_name]
	reference = image_captions_pair[img_name]
	reference_final=[]
	for ref_list in reference:
			reference_final.append(ref_list.split())
#	print(reference_final)
	hypotheses_spice.append(hypothesis)
	references_spice.append(reference)

	hypothesis = hypothesis.split()
	# bleus.append(nltk.translate.bleu_score.sentence_bleu(reference_final,hypothesis,weights=[1]))
	hypotheses.append(hypothesis)
	references.append(reference_final)
	

bleuScore1 = compute_bleu(references,hypotheses,max_order=1)[0]
bleuScore2 = compute_bleu(references,hypotheses,max_order=2)[0]
bleuScore3 = compute_bleu(references,hypotheses,max_order=3)[0]
bleuScore4 = compute_bleu(references,hypotheses,max_order=4)[0]

nlgeval = NLGEval(no_skipthoughts=True,no_glove=True)  # loads the models
metrics_dict = nlgeval.compute_metrics(references_spice, hypotheses_spice)
print metrics_dict

print "BLEU-1 Score of the Model: ", bleuScore1
print "BLEU-2 Score of the Model: ", bleuScore2
print "BLEU-3 Score of the Model: ", bleuScore3
print "BLEU-4 Score of the Model: ", bleuScore4
