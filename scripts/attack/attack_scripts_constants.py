checkpoint_strategies2checkpoint_strings_llama_secalign = {
    # all_checkpoints
    "all_checkpoints": """checkpoints=`seq 0 897`
echo "Checkpoints: ${checkpoints[@]}"
    """,
    "step": """checkpoints1=$(seq 0 30)
checkpoints2=$(seq 50 50 898)
checkpoints3=897
checkpoints=($checkpoints1 $checkpoints2 $checkpoints3)
echo "Checkpoints: ${checkpoints[@]}"
    """,
    "loss": """checkpoints=(0   1   3   4   5   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  24  25  26  27  29  31  32  33  36  37  38  40  41  45  46  49  50  59  60  61  65  66  72  74  77  78  85  91  92  95  96  98  99 102 103 104 109 111 118 121 125 126 131 134 152 153 154 159 160 162 164 165 166 171 172 174 175 177 178 189 190 191 192 200 201 219 220 222 223 224 225 226 238 239 267 268 274 275 277 278 280 281 285 286 336 337 338 388 432 433 456 457 507 557 607 657 707 757 776 777 827 877 897)
echo "Checkpoints: ${checkpoints[@]}"
    """,
    "gradnorm": """checkpoints=(0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  37  38  40  41  43  45  46  49  54  58  59  60  65  72  77  78  79  80  81  84  91  98 102 103 104 109 110 117 118 119 120 125 152 153 157 159 162 163 165 171 174 177 186 189 191 200 219 222 223 225 235 238 257 265 267 274 277 280 285 287 382 432 456 464 484 897)
echo "Checkpoints: ${checkpoints[@]}"
""",
#     # gcg adversarial finetuning checkpoints
#     "gradnorm": """checkpoints=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)
# echo "Checkpoints: ${checkpoints[@]}"
# """,
    "everyk10": """checkpoints1=$(seq 0 10 898)
checkpoints2=897
checkpoints=($checkpoints1 $checkpoints2)
echo "Checkpoints: ${checkpoints[@]}"
    """,
    "everyk50": """checkpoints1=$(seq 0 50 898)
checkpoints2=897
checkpoints=($checkpoints1 $checkpoints2)
echo "Checkpoints: ${checkpoints[@]}"
    """,
    "everyk100": """checkpoints1=$(seq 0 100 898)
checkpoints2=897
checkpoints=($checkpoints1 $checkpoints2)
echo "Checkpoints: ${checkpoints[@]}"
    """,
    "stepq10": """checkpoints1=$(seq 0 30)
checkpoints2=$(seq 40 10 898)
checkpoints3=897
checkpoints=($checkpoints1 $checkpoints2 $checkpoints3)
echo "Checkpoints: ${checkpoints[@]}"
    """,
    "stepq100": """checkpoints1=$(seq 0 30)
checkpoints2=$(seq 100 100 898)
checkpoints3=897
checkpoints=($checkpoints1 $checkpoints2 $checkpoints3)
echo "Checkpoints: ${checkpoints[@]}"
    """,
    "gradnormt01": """checkpoints=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 30 31 32 34 37 40 46 49 59 60 65 77 91 102 103 118 119 120 152 153 159 174 177 189 191 200 219 222 267 277 280 285 432 484 897)
echo "Checkpoints: ${checkpoints[@]}"
""",
    "gradnormt02": """checkpoints=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 31 40 49 60 91 102 103 118 119 189 219 222 277 432 484 897)
echo "Checkpoints: ${checkpoints[@]}"
""",
    "gradnormt001": """checkpoints=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 53 54 55 57 58 59 60 62 63 64 65 66 67 68 69 70 71 72 73 74 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 94 95 97 98 99 102 103 104 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 122 124 125 127 128 129 130 131 132 133 135 136 139 140 142 144 145 147 148 149 150 151 152 153 155 157 159 162 163 164 165 168 170 171 174 175 177 182 183 184 186 187 189 191 194 200 201 203 208 209 219 220 222 223 225 229 230 231 232 233 235 236 238 239 240 241 242 243 244 245 246 247 248 251 254 257 258 259 260 262 263 265 267 274 277 280 285 287 288 295 302 317 318 320 325 334 341 351 359 365 382 387 388 394 399 405 413 427 432 451 453 455 456 464 466 476 477 480 484 486 499 501 520 522 523 537 580 585 586 625 627 688 729 760 763 775 849 869 871 893 897)
echo "Checkpoints: ${checkpoints[@]}"
""",
}

checkpoint_strategies2checkpoint_strings_mistral_secalign = {
    "step": """checkpoints1=$(seq 0 30)
checkpoints2=$(seq 50 50 898)
checkpoints3=897
checkpoints=($checkpoints1 $checkpoints2 $checkpoints3)
echo "Checkpoints: ${checkpoints[@]}"
    """,
    "gradnorm": """checkpoints=(0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19   20   21   22   24   25   27   28   29   30   31   32   33   34   36   37   38   39   44   48   50   63   68   69   76   78   84   90   91   106   109   111   124   133   136   139   155   164   167   171   173   174   188   198   201   215   222   235   236   242   250   253   265   266   267   290   299   303   304   329   529   539   540   555   559   580   585   588   594   638   818   832   866   897)
echo "Checkpoints: ${checkpoints[@]}"
""",
}

checkpoint_strategies2checkpoint_strings_qwen_secalign = {
    "gradnorm": """checkpoints=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 50 52 54 55 71 72 74 80 83 98 101 104 105 106 121 132 135 136 143 150 163 178 186 187 190 191 206 208 223 226 238 288 313 338 369 374 458 499 517 596 801 870 897)
echo "Checkpoints: ${checkpoints[@]}"
""",
}

checkpoint_strategies2checkpoint_strings_llama_struq = {
    "gradnorm": """checkpoints=(0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   30   31   33   35   36   41   42   43   47   50   63   96  100  104  180  197  220  281  282  287  325  379  407  437  479  593  652  695  711  727  777  842  963 1004 1017 1045 1064 1109 1152 1165 1218 1281 1377 1504 1546 1607 1662 1665 1688 1701 1705 1730 1731 1739 1739 1757 1778 1793 1830 1849 1859 1872 1887 1901 1910 1938 1953 1962 1964 1997 2000 2047 2055 2065 2084 2087 2102 2104 2117 2119 2135 2139 2153 2156 2177 2196 2209 2227 2251 2286 2299 2301 2319 2320 2356 2360 2371 2421 2424)
echo "Checkpoints: ${checkpoints[@]}"
"""
}

checkpoint_strategies2checkpoint_strings_mistral_struq = {
    "gradnorm": """checkpoints=(0 1 2 3 4 5 6 7 8 9 11 13 20 24 30 35 43 47 50 87 94 111 158 180 189 194 221 236 247 268 274 281 290 300 306 316 319 325 378 398 400 403 407 413 418 435 465 471 474 476 499 501 521 526 537 539 563 582 593 624 636 652 687 689 690 731 741 762 765 777 807 826 842 853 886 887 1015 1017 1029 1064 1101 1109 1165 1204 1210 1220 1252 1281 1314 1320 1372 1388 1393 1443 1452 1456 1474 1522 1583 1591 1601 1612 1690 1695 1731 1765 1778 1836 1962 1964 2418 2424)
echo "Checkpoints: ${checkpoints[@]}"
"""
}

checkpoint_strategies2checkpoint_strings_qwen_struq = {
    "gradnorm": """checkpoints=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 30 31 32 33 36 39 40 41 42 43 44 47 48 50 51 52 54 56 60 61 63 69 94 95 96 97 100 104 115 116 119 128 154 155 158 162 175 180 181 184 190 193 211 212 217 223 227 236 270 273 274 287 299 303 400 435 474 626 652 689 695 778 878 1017 1477 1872 2106 2209 2258 2319 2421 2424)
echo "Checkpoints: ${checkpoints[@]}"
""",
}

checkpoint_strategies2checkpoint_strings_safety_llama = {
    "gradnorm": """checkpoints=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 51 53 55 168 229 270 303 322 335 366 367 389 393 395 453 482 486 502 513 519 525 526 555 562 569 577 589 592 593 598 600 603 611 620 629 631 637 645 656 666 668)
echo "Checkpoints: ${checkpoints[@]}"
"""
}

hpc_header = """#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=1:mem=30gb:ngpus=1

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate secalign
cd project_dir
"""

checkpoint_attack_bash_script = """
current_time="{current_time}"
echo "Current time: $current_time"

# Description:{description}

for checkpoint in "${{checkpoints[@]}}"
do
    python test.py \\
        --model_name_or_path "{model_path}" \\
        --device "{device}" \\
        --defense "{defense}" \\
        --data_path "{data_path}" \\
        --checkpoint_dir "{checkpoint_dir}" \\
        --checkpoint $checkpoint \\
        --all_checkpoints ${{checkpoints[@]}} \\
        --checkpoint_choice "{checkpoint_choice}" \\
        --current_time $current_time \\
        --sample_ids {sample_ids} \\
        --gcg_global_budget \\
        --gcg_early_stopping \\
        {universal_or_individual_attack_args} \\
        {additional_args}
done
"""

direct_attack_bash_script = """
current_time="{current_time}"
echo "Current time: $current_time"

# Description:
{description}

python test.py \\
    --model_name_or_path "{model_path}" \\
    --device "{device}" \\
    --defense "{defense}" \\
    --data_path "{data_path}" \\
    --checkpoint_dir "{checkpoint_dir}" \\
    --current_time $current_time \\
    --sample_ids {sample_ids} \\
    --gcg_global_budget \\
    --gcg_early_stopping \\
    {universal_or_individual_attack_args} \\
    {additional_args}
"""

individual_sample_attack_args = (
    """--gcg_num_steps_per_checkpoint {gcg_num_steps_per_checkpoint} --gcg_num_steps_total {gcg_num_steps_total}"""
)

universal_attack_args = """--gcg_num_steps_per_sample 3500 --gcg_num_steps_total 10000 --gcg_universal_attack"""