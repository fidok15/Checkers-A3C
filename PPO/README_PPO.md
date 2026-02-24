potem to dokładniej tu opiszę,

Odpalanie: Puszczamy plik train.py z tego folderu, jak juz mamy jakis gotowy model z poprzedniego przerwanego treningu to bierzemy ostatni model z folderu checkpoints i wrzucamy do terminala 
python train.py --resume checkpoints/ppo_checkers_200.pt --total-steps 5000000

Granie:
..\.venv\Scripts\python.exe play.py --checkpoint checkpoints\ppo_checkers_1100.pt

spróbujcie ogarnąć venva i komenda w terminalu jak wyżej, wybieracie checkpoint i sobie z nim gracie w konsoli, pewnie jak powiecie copilotowi to wam ogarnie