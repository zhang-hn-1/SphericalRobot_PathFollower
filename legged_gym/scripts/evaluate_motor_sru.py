"""Fixed motor-policy evaluation; same implementation and GPU guard as training."""
import sys
from train_motor_sru import main


if __name__ == '__main__':
    if '--mode' in sys.argv[1:]:
        raise SystemExit('evaluate_motor_sru.py fixes --mode eval; omit --mode')
    main(['--mode', 'eval'] + sys.argv[1:])
