### Dataset
# ESMFold & AlphaFold2: DeepLocBinary DeepLocMulti MetalIonBinding EC Thermostability
# ESMFold: DeepSol DeepSoluE
# No structure: FLIP_AAV FLIP_GB1

### Protein Language Model (PLM)
# facebook: esm2_t30_150M_UR50D esm2_t33_650M_UR50D esm2_t36_3B_UR50D
# RostLab: prot_bert prot_bert_bfd prot_t5_xl_uniref50 prot_t5_xl_bfd ankh-base ankh-large
export HF_ENDPOINT=https://hf-mirror.com
dataset=DeepLocBinary
pdb_type=AlphaFold2
plm_source=facebook
plm_model=esm2_t30_150M_UR50D
lr=5e-4
python src/train.py \
    --plm_model $plm_source/$plm_model \
    --dataset_config data/$dataset/"$dataset"_"$pdb_type"_HF.json \
    --learning_rate $lr \
    --num_epochs 50 \
    --batch_token 12000 \
    --gradient_accumulation_steps 8 \
    --patience 3 \
    --structure_seq foldseek_seq,ss8_seq \
    --output_dir debug/$dataset/$plm_model \
    --output_model_name ses-adapter_"$pdb_type"_lr"$lr"_bt12k_ga8.pt
