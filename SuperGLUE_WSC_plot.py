import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import spearmanr

def PlotHeatMap(ax,plot_data,title_text):
    im = ax.imshow(plot_data.mean(axis=0))
    fig.colorbar(im, ax=ax,shrink=0.7)
    ax.set_title(f'{title_text}\n{plot_data.mean(axis=0).max(axis=-1)[-1]:.5f}')

def PlotLine(ax,plot_data):
    ax.plot(plot_data.mean(axis=0).max(axis=-1),np.arange(plot_data.shape[1]))
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, required = True)
    parser.add_argument('--roll_out',dest='roll_out',action='store_true')
    parser.add_argument('--context_attn_type',choices=['pron','mean'],default='pron')
    parser.add_argument('--input_type',choices=['','_masked'],default='')
    parser.add_argument('--strict',dest='strict',action='store_true')
    parser.add_argument('--stimuli', type = str,
                        choices=['original','control_gender','control_number','control_combined'],
                        default='original')
    parser.set_defaults(roll_out=False)
    parser.set_defaults(strict=False)
    args = parser.parse_args()

    if args.roll_out:
        roll_out_id = '_roll_out'
    else:
        roll_out_id = ''

    if args.strict:
        strict_id = '_strict'
    else:
        strict_id = ''

    # load the data
    with open(f'datafile/superglue_wsc_prediction_{args.model}_{args.stimuli}.pkl','rb') as f:
        pred_data = pickle.load(f)
    with open(f'datafile/superglue_wsc_attention_{args.model}_{args.stimuli}{roll_out_id}.pkl','rb') as f:
        attn_data = pickle.load(f)

    # calculate the prediction score on the sentence level
    pred_score = []
    for pair_id,data in pred_data.items():
        pred_score.extend([data['ave_1'][0]-data['ave_1'][1],data['ave_2'][0]-data['ave_2'][1]])
    pred_score = np.array(pred_score)
    print(f'prediction score of {args.model}: {np.mean(pred_score>0)}')

    # calculate the attention score on the sentence level
    attn_score = []
    for pair_id,data in attn_data.items():
        attn_score.extend([data['choice_1'][0]-data['choice_1'][1],data['choice_2'][0]-data['choice_2'][1]])
    attn_score = np.array(attn_score)
    print(f'maximum attention score for {args.model}: {np.mean(attn_score>0,axis=0).max()}')


    # calculate the prediction score on the sentence pair level
    paired_pred_score = []
    for pair_id,data in pred_data.items():
        paired_pred_score.append((data['ave_1'][0]-data['ave_1'][1])+(data['ave_2'][0]-data['ave_2'][1]))
    paired_pred_score = np.array(paired_pred_score)
    print(f'paired prediction score of {args.model}: {np.mean(paired_pred_score>0)}')

    # calculate the STRICT prediction score on the sentence pair level
    strict_paired_pred_score = []
    for pair_id,data in pred_data.items():
        strict_paired_pred_score.append(data['ave_1'][0]>data['ave_1'][1] and data['ave_2'][0]>data['ave_2'][1])
    strict_paired_pred_score = np.array(strict_paired_pred_score)
    print(f'strict paired prediction score of {args.model}: {np.mean(strict_paired_pred_score==1)}')

    # calculate the attention score on the sentence pair level
    # overall attention on candidate noun phrases
    paired_attn_score_choices_all = []
    for pair_id,data in attn_data.items():
        paired_attn_score_choices_all.append((data[f'choice{args.input_type}_1'][0]+data[f'choice{args.input_type}_1'][1]+
                                              data[f'choice{args.input_type}_2'][0]+data[f'choice{args.input_type}_2'][1])/4)
    paired_attn_score_choices_all = np.array(paired_attn_score_choices_all)

    # overall attention on the correct noun phrases
    paired_attn_score_choices_correct = []
    for pair_id,data in attn_data.items():
        paired_attn_score_choices_correct.append((data[f'choice{args.input_type}_1'][0]+data[f'choice{args.input_type}_2'][0])/2)
    paired_attn_score_choices_correct = np.array(paired_attn_score_choices_correct)

    # overall attention on the incorrect noun phrases
    paired_attn_score_choices_incorrect = []
    for pair_id,data in attn_data.items():
        paired_attn_score_choices_incorrect.append((data[f'choice{args.input_type}_1'][1]+data[f'choice{args.input_type}_2'][1])/2)
    paired_attn_score_choices_incorrect = np.array(paired_attn_score_choices_incorrect)

    #  differential attention on the correct referents
    paired_attn_score_choices = []
    for pair_id,data in attn_data.items():
        paired_attn_score_choices.append((data[f'choice{args.input_type}_1'][0]-data[f'choice{args.input_type}_1'][1])+
                                         (data[f'choice{args.input_type}_2'][0]-data[f'choice{args.input_type}_2'][1]))
    paired_attn_score_choices = np.array(paired_attn_score_choices)

    # overall attention on context words
    paired_attn_score_context = []
    for pair_id,data in attn_data.items():
        paired_attn_score_context.append((data[f'context{args.input_type}_1'][args.context_attn_type]+
                                          data[f'context{args.input_type}_2'][args.context_attn_type])/2)
    paired_attn_score_context = np.array(paired_attn_score_context)

    # overall attention on periods
    paired_attn_score_period = []
    for pair_id,data in attn_data.items():
        paired_attn_score_period.append((data[f'period{args.input_type}_1']+data[f'period{args.input_type}_2'])/2)
    paired_attn_score_period = np.array(paired_attn_score_period)

    # calculate correlation of the attention metric with the prediction metric
    paired_attn_choices_all_corr = np.array([[spearmanr(paired_attn_score_choices_all[:,layer_id,head_id],paired_pred_score)[0]
                                              for head_id in range(attn_score.shape[2])] for layer_id in range(attn_score.shape[1])])
    paired_attn_choices_corr = np.array([[spearmanr(paired_attn_score_choices[:,layer_id,head_id],paired_pred_score)[0]
                                          for head_id in range(attn_score.shape[2])] for layer_id in range(attn_score.shape[1])])
    paired_attn_context_corr = np.array([[spearmanr(paired_attn_score_context[:,layer_id,head_id],paired_pred_score)[0]
                                          for head_id in range(attn_score.shape[2])] for layer_id in range(attn_score.shape[1])])

    # plot the attentions
    if args.model in ['bert-base-uncased','roberta-base']:
        fig,axs = plt.subplots(1,10,figsize=(25,5),dpi=120,gridspec_kw={'width_ratios': [1,7,1,7,1,7,1,7,1,7]})
    elif args.model in ['bert-large-cased','roberta-large']:
        fig,axs = plt.subplots(1,10,figsize=(25,5),dpi=120,gridspec_kw={'width_ratios': [1,9,1,9,1,9,1,9,1,9]})

    '''
    PlotLine(axs[0],paired_attn_score_choices_all)
    PlotHeatMap(axs[1],paired_attn_score_choices_all,'overall attention on NPs')
    '''

    PlotLine(axs[0],paired_attn_score_choices_correct)
    PlotHeatMap(axs[1],paired_attn_score_choices_correct,'overall attention on correct NPs')

    PlotLine(axs[2],paired_attn_score_choices_incorrect)
    PlotHeatMap(axs[3],paired_attn_score_choices_incorrect,'overall attention on incorrect NPs')

    PlotLine(axs[4],paired_attn_score_context)
    PlotHeatMap(axs[5],paired_attn_score_context,'attention on context words')

    PlotLine(axs[6],paired_attn_score_choices)
    PlotHeatMap(axs[7],paired_attn_score_choices,'attention difference')

    PlotLine(axs[8],paired_attn_score_period)
    PlotHeatMap(axs[9],paired_attn_score_period,'attention on periods')

    plt.tight_layout()
    fig.savefig(f'figures/superglue_wsc_attn_all_{args.model}_{args.context_attn_type}_{args.stimuli}{args.input_type}{roll_out_id}.png')

    # plot the attentions
    if args.model in ['bert-base-uncased','roberta-base']:
        fig,axs = plt.subplots(2,5,figsize=(25,10),dpi=120)
    elif args.model in ['bert-large-cased','roberta-large']:
        fig,axs = plt.subplots(2,5,figsize=(25,10),dpi=120)

    if args.strict:
        PlotHeatMap(axs[0,0],paired_attn_score_choices_correct[strict_paired_pred_score==1],'overall attention on correct NPs')
        axs[0,0].set_ylabel('Correct Prediction')
        PlotHeatMap(axs[0,1],paired_attn_score_choices_incorrect[strict_paired_pred_score==1],'overall attention on incorrect NPs')
        PlotHeatMap(axs[0,2],paired_attn_score_context[strict_paired_pred_score==1],'attention on context words')
        PlotHeatMap(axs[0,3],paired_attn_score_choices[strict_paired_pred_score==1],'attention difference')
        PlotHeatMap(axs[0,4],paired_attn_score_period[strict_paired_pred_score==1],'attention on periods')

        PlotHeatMap(axs[1,0],paired_attn_score_choices_correct[strict_paired_pred_score!=1],'overall attention on correct NPs')
        axs[1,0].set_ylabel('Incorrect Prediction')
        PlotHeatMap(axs[1,1],paired_attn_score_choices_incorrect[strict_paired_pred_score!=1],'overall attention on incorrect NPs')
        PlotHeatMap(axs[1,2],paired_attn_score_context[strict_paired_pred_score!=1],'attention on context words')
        PlotHeatMap(axs[1,3],paired_attn_score_choices[strict_paired_pred_score!=1],'attention difference')
        PlotHeatMap(axs[1,4],paired_attn_score_period[strict_paired_pred_score!=1],'attention on periods')

    else:
        PlotHeatMap(axs[0,0],paired_attn_score_choices_correct[paired_pred_score>0],'overall attention on correct NPs')
        axs[0,0].set_ylabel('Correct Prediction')
        PlotHeatMap(axs[0,1],paired_attn_score_choices_incorrect[paired_pred_score>0],'overall attention on incorrect NPs')
        PlotHeatMap(axs[0,2],paired_attn_score_context[paired_pred_score>0],'attention on context words')
        PlotHeatMap(axs[0,3],paired_attn_score_choices[paired_pred_score>0],'attention difference')
        PlotHeatMap(axs[0,4],paired_attn_score_period[paired_pred_score>0],'attention on periods')

        PlotHeatMap(axs[1,0],paired_attn_score_choices_correct[paired_pred_score<=0],'overall attention on correct NPs')
        axs[1,0].set_ylabel('Incorrect Prediction')
        PlotHeatMap(axs[1,1],paired_attn_score_choices_incorrect[paired_pred_score<=0],'overall attention on incorrect NPs')
        PlotHeatMap(axs[1,2],paired_attn_score_context[paired_pred_score<=0],'attention on context words')
        PlotHeatMap(axs[1,3],paired_attn_score_choices[paired_pred_score<=0],'attention difference')
        PlotHeatMap(axs[1,4],paired_attn_score_period[paired_pred_score<=0],'attention on periods')

    plt.tight_layout()
    fig.savefig(f'figures/superglue_wsc_attn_all_indiv_{args.model}_{args.context_attn_type}_{args.stimuli}{args.input_type}{roll_out_id}{strict_id}.png')

    # plot correlations
    if args.model in ['bert-base-uncased','roberta-base']:
        fig,axs = plt.subplots(1,6,figsize=(15,5),dpi=120,gridspec_kw={'width_ratios': [1,7,1,7,1,7]})
    elif args.model in ['bert-large-cased','roberta-large']:
        fig,axs = plt.subplots(1,6,figsize=(15,5),dpi=120,gridspec_kw={'width_ratios': [1,9,1,9,1,9]})

    axs[0].plot(paired_attn_choices_all_corr.max(axis=-1),np.arange(attn_score.shape[1]))
    axs[0].invert_xaxis()
    axs[0].invert_yaxis()
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["bottom"].set_visible(False)
    axs[0].spines["left"].set_visible(False)
    axs[0].spines["right"].set_visible(False)

    im = axs[1].imshow(paired_attn_choices_all_corr)
    fig.colorbar(im, ax=axs[1],shrink=0.7)
    axs[1].set_title('overall attention on noun phrases')

    axs[2].plot(paired_attn_context_corr.max(axis=-1),np.arange(attn_score.shape[1]))
    axs[2].invert_xaxis()
    axs[2].invert_yaxis()
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["bottom"].set_visible(False)
    axs[2].spines["left"].set_visible(False)
    axs[2].spines["right"].set_visible(False)

    im = axs[3].imshow(paired_attn_context_corr)
    fig.colorbar(im, ax=axs[3],shrink=0.7)
    axs[3].set_title('attention on context words')

    axs[4].plot(paired_attn_choices_corr.max(axis=-1),np.arange(attn_score.shape[1]))
    axs[4].invert_xaxis()
    axs[4].invert_yaxis()
    axs[4].spines["top"].set_visible(False)
    axs[4].spines["bottom"].set_visible(False)
    axs[4].spines["left"].set_visible(False)
    axs[4].spines["right"].set_visible(False)

    im = axs[5].imshow(paired_attn_choices_corr)
    fig.colorbar(im, ax=axs[5],shrink=0.7)
    axs[5].set_title('attention difference')

    plt.tight_layout()
    fig.savefig(f'figures/corr/superglue_wsc_attn_corr_all_{args.model}_{args.context_attn_type}_{args.stimuli}{args.input_type}{roll_out_id}.png')
