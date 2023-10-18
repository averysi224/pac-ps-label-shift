GPU_ID=2,3
EPS=0.03
DELTA=5e-4
DELTAIW=5e-4
DELTACAL=5e-4

M=50000

SRC=Entity
TAR=Entity

EXPNAME=Entity

MDLPATH=snapshots_models/Entity/model_params_best
SDMDLPATH=snapshots_models/Entity/model_params_srcdisc_best

for i in {1..100}
do
	PS
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_twonormals.py \
			--exp_name exp_${EXPNAME}_naive_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method pac_predset_CP \
			--data.seed None \
			--model_predset.eps $EPS \
			--model_predset.delta $DELTA \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH  
    # ### PS-R
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_twonormals.py \
			--exp_name exp_${EXPNAME}_rejection_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method pac_predset_rejection \
			--data.seed None \
			--model_predset.eps $EPS \
			--model_predset.delta $DELTA \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH \
			--model_sd.path_pretrained $SDMDLPATH
    # # PS-W
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_twonormals.py \
			--exp_name exp_${EXPNAME}_worst_rejection_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method pac_predset_worst_rejection \
			--data.seed None \
			--model_predset.eps $EPS \
			--model_predset.delta $DELTA \
			--model_iwcal.delta $DELTA \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH \
			--model_sd.path_pretrained $SDMDLPATH 
	# # maxiw
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_twonormals.py \
			--exp_name exp_${EXPNAME}_maxiw_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method pac_ps_maxiw \
			--data.seed None \
			--model_predset.eps $EPS \
			--model_predset.delta $DELTA \
			--model_iwcal.delta $DELTA \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH \
			--model_sd.path_pretrained $SDMDLPATH 
	# WCP
	CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_twonormals.py \
			--exp_name exp_${EXPNAME}_cp_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method cp_ls \
			--data.seed None \
			--model_predset.eps $EPS \
			--model_predset.delta $DELTA \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH \
			--model_sd.path_pretrained $SDMDLPATH
done 

