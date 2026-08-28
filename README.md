# Auditing Prompt Injection Defenses Before They Reach Users: What Strong Adversaries Reveal
This repository contains the source code for the paper "Auditing Prompt Injection Defenses Before They Reach Users: What Strong Adversaries Reveal" by Xiaoxue Yang*, Bozhidar Stevanoski*, Matthieu Meeus, and Yves-Alexandre de Montjoye (* denotes equal contribution).

## Abstract
Large language models are increasingly deployed in user-facing products that handle sensitive personal data while interacting with untrusted external content. This makes them a prime target for prompt injection attacks, in which an adversary injects malicious instructions in that content to hijack the model. Fine-tuning-based defenses have been proposed to mitigate these attacks by training models to follow benign user instructions and ignore injected ones. Lacking formal guarantees, these defenses are evaluated empirically against the most effective realistic attacks available at the time, often reporting near-zero attack success rates -- giving developers, and the users who depend on them, a false sense of security and leaving open the possibility that future, more capable attacks could circumvent them and cause severe consequences in widely deployed systems. In this paper, we argue that developers should stress-test their defenses before release, and we introduce an auditing framework that does so without requiring defenders to invent the next state-of-the-art attack. Our key insight is that the fine-tuning process itself yields useful artifacts: intermediate checkpoints that serve as stepping stones for optimization-based attacks. Although these artifacts are unavailable to real attackers, we show that the vulnerabilities they expose are genuine and can be exploited by realistic adversaries. Instantiating our framework with optimization-based attacks such as GCG and BEAST, we raise attack success rates against state-of-the-art defenses from near zero to as high as 96\%, revealing weaknesses missed by prior evaluations. Our results show that current evaluation practices can substantially overestimate robustness, and that auditing defenses under strong adversaries before deployment shortens the feedback loop and helps prevent attacks from causing real-world harm to users.


## Environment setup
+ Install environment dependencies

    ```
    conda create -n checkpoint_attack python==3.10 
    ```

+ Install package dependencies

  + For finetuning and attacking using SecAlign and StruQ (we adopted the requirements in `requirements.txt` in [SecAlign](https://github.com/facebookresearch/SecAlign/tree/main)):

    ```
    pip install -r requirements_secalign_struq.txt
    ```

  + For finetuning and attacking using Safety-Tuned LLaMAs (we used the `requirements.txt` from [Safety-Tuned LLaMAs](https://github.com/vinid/safety-tuned-llamas) and installed the listed packages with their latest available versions):

    ```
    pip install -r requirements_safety_tuned_llama.txt
    ```

+ Download data dependencies

    ```
    python setup.py
    ```


## SecAlign 
+ To finetune Llama3-8B-Instruct, Mistral-7B-Instruct, and Qwen2-1.5B-Instruct using SecAlign, run the following respective commands:
  ```
    bash scripts/defense/secalign_llama3instruct.sh
    bash scripts/defense/secalign_mistralinstruct.sh
    bash scripts/defense/secalign_qwen.sh
  ```

## StruQ
+ Similarly, to finetune Llama3-8B-Instruct, Mistral-7B-Instruct, and  Qwen2-1.5B-Instruct using StruQ, run the following respective commands:
  ```
    bash scripts/defense/struq_llama3instruct.sh
    bash scripts/defense/struq_mistralinstruct.sh
    bash scripts/defense/struq_qwen.sh
  ```


## Safety-Tuned LLaMAs
+ To finetune for Safety-Tuned LLaMAs, run the following script, which uses `data/training/saferpaca_Instructions_2000.json` formatted with `data/configs/alpaca.json` as training data. 
    ```
    python safety_llama_finetuning.py
    ```

## Test
+ To run standard and Checkpoint attacks (GCG and BEAST) against defense(s) and model(s), run the following to automatically generate attack `.sh` scripts:

    ```
    python scripts/attack/generate_attack_scripts.py
    ```
+ Run the relevant `.sh` script(s) in `scripts/attack` to launch the desired attacks:

    + Standard vs Checkpoint attack
        + Standard attack shell scripts (directly attacking the final finetuned model $\theta_C$) have "direct" in the script filenames 
        + Checkpoint attack shell scripts have "checkpoint" in the script filenames, as well as the appropriate checkpoint selection strategy
    + Attacking individual samples vs universal attack
        + Individual-sample attack shell scripts have "individual" in the script filenames
        + Universal attack shell scripts have "universal" in the script filenames 

This repository is licensed under the MIT License. It builds on [SecAlign](https://github.com/facebookresearch/SecAlign/tree/main) (CC BY-NC 4.0), [Safety-Tuned LLaMAs](https://github.com/vinid/safety-tuned-llamas) (MIT), [Attention-Tracker](https://github.com/khhung-906/Attention-Tracker) (CC BY-NC 4.0), and [BEAST](https://github.com/vinusankars/BEAST). We thank the authors for open-sourcing their work.
