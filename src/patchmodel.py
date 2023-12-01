#!/usr/bin/python3
'''
Apply patches to serialized models.
'''

import argparse
from am.am import PQHDC, AMMap, AMSignQuantize, AMThermometerDeviation
from hdc_models import *
import sys

import common

patcheable_ams = {
        'TQHD': AMThermometerDeviation,
        'SQ': AMSignQuantize,
        'PQHDC': PQHDC,
        }

def patch_to(am_class, *args, **kwargs):
    f = {
            AMThermometerDeviation: patch_to_AMThermometerDeviation,
            AMSignQuantize: patch_to_AMSignQuantize,
            PQHDC: patch_to_PQHDC,
            }
    return f[am_class](am_class, *args, **kwargs)

def patch_to_AMThermometerDeviation(am_class, args, am, **kwargs):
    return am_class.from_AMMap(
            am,
            bits=args.am_bits,
            intervals=args.am_intervals,
            deviation=args.am_tqhd_deviation,
            **kwargs,
        )

def patch_to_AMSignQuantize(am_class, args, am, **kwargs):
    return am_class.from_AMMap(
            am,
            **kwargs
        )

def patch_to_PQHDC(am_class, args, am, **kwargs):
    return am_class.from_AMMap(
            am,
            projections=args.am_pqhdc_projections,
            **kwargs
        )

def transform_am(args, model):
    """docstring for transform_am"""
    if args.am_type not in patcheable_ams:
        raise RuntimeError(f'Attempt to patch AM to unsupported type {args.am_type}.')

    if not isinstance(model.am, AMMap):
        raise RuntimeError('This script only works with AMMap models.')

    model_am = model.am

    target_class = patcheable_ams[args.am_type]
    prediction = common.args_pick_prediction(args)
    # Only transform AM type if they are not the same type
    if target_class is not type(model_am):
        new_am = patch_to(target_class, args, model_am, prediction=prediction)
    # If they are, change only the prediction strategy
    else:
        new_am = model_am
        new_am.prediction = prediction

    model.am = new_am
    return model

def main():
    parser = argparse.ArgumentParser(description='Model patcher.')
    parser.add_argument(
            'model',
            help='path to original model.',
        )
    parser.add_argument(
            'path',
            help='Path to patched model. Parent folders are created if necessary.',
        )

    common.add_am_arguments(parser)

    args = parser.parse_args()

    # TODO: This script needs better arument parsing
    # Workaround the reuse of am_arguments in common.
    model = common.load_model(args.model)
    if '--am-type' in sys.argv:
        model = transform_am(args, model)
    elif '--am-prediction':
        prediction = common.args_pick_prediction(args)
        model = common.load_model(args.model)
        model.am.prediction = prediction

    common.save_model(model, args.path)

if __name__ == '__main__':
    main()
