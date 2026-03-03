potem to dokładniej tu opiszę,

Tipy:
domyślnie komendy działają kiedy działamy w folderze PPO (cd PPO) ale pewnie gdzieś o tym pozapominałem wiec trzeba uważać na ścieżki

Trenowanie: Puszczamy plik train.py z tego folderu, jak juz mamy jakis gotowy model z poprzedniego przerwanego treningu to bierzemy ostatni model z folderu checkpoints i wrzucamy do terminala np:
python train.py --resume checkpoints/ppo_checkers_200.pt --total-steps 5000000
najważniejsze parametry do treningu:
-- lr : learning rate
-- total-steps : ile maksymalnie ruchów może być w jednej grze
-- rollout-steps : co ile ruchów model aktualizuje wagi
-- resume : wznawia trenowanie zapisanego modelu (po resume dajemy path do modelu)
-- mcts-ratio : jaki procent przeciwników to mcts (domyślnie 30%)
-- mcts-playout : jaka głębokość mcts (domyślnie 100)

Gra na mcts:
cd PPO
python evaluate.py --checkpoint checkpoints/ppo_checkers_1100.pt 
parametry do usatwiania:
-- games : liczba gier (jako oba kolory więc 2 razy więcej niż wpiszemy)
-- checkpoint : wpisujemy model do przetestowania (obecnie wszystkie daje do folderu checkpoint)
-- mcts-playouts : jak dużo ścieżek sprawdza mcts
-- max-steps : limit ruchów na grę
-- gpu : używamy gpu jeśli mamy

Granie:
..\.venv\Scripts\python.exe play.py --checkpoint checkpoints\ppo_checkers_1100.pt

