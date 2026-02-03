#!/usr/bin/env python3
from math import fabs, exp
import numpy as np

from opendbc.car import get_safety_config, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.gm.carcontroller import CarController
from opendbc.car.gm.carstate import CarState
from opendbc.car.gm.radar_interface import RadarInterface, RADAR_HEADER_MSG, CAMERA_DATA_HEADER_MSG
from opendbc.car.gm.values import CAR, CarControllerParams, EV_CAR, CAMERA_ACC_CAR, SDGM_CAR, ALT_ACCS, CanBus, GMSafetyFlags
from opendbc.car.interfaces import CarInterfaceBase, TorqueFromLateralAccelCallbackType, LateralAccelFromTorqueCallbackType

TransmissionType = structs.CarParams.TransmissionType
NetworkLocation = structs.CarParams.NetworkLocation

# Acadia ported from siglin
# Silverado and Bolt are curves fit to steering data
PL_TORQUE_PARAMS = {
  CAR.GMC_ACADIA: (
    [-4.        , -3.57894737, -3.15789474, -2.73684211, -2.31578947,
        -1.89473684, -1.47368421, -1.05263158, -0.63157895, -0.21052632,
         0.21052632,  0.63157895,  1.05263158,  1.47368421,  1.89473684,
         2.31578947,  2.73684211,  3.15789474,  3.57894737,  4.        ],
         [-1.69288228, -1.56142961, -1.42997674, -1.2985223 , -1.16705617,
        -1.03550255, -0.90329485, -0.76622793, -0.59468496, -0.24210848,
         0.35394392,  0.7065204 ,  0.87806337,  1.01513029,  1.14733799,
         1.27889161,  1.41035774,  1.54181218,  1.67326505,  1.80471772]
  ),
  CAR.CHEVROLET_SILVERADO: (
    [-1.94345798, -1.36092958, -1.1403284 , -0.86851078, -0.62763315,
        -0.42766058, -0.30679224, -0.24618561, -0.20313874, -0.16282139,
        -0.11081693, -0.05341054,  0.04738075,  0.18307001,  0.28958397,
         0.39225609,  0.53721741,  0.72810697,  0.92637494,  1.16260609,
         1.7156973 ],
         [-1. , -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0. ,
         0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]
  ),
  CAR.CHEVROLET_BOLT_EUV:
      ([-2.26916035, -1.51287849, -1.36134079, -1.06070991, -0.7886563 ,
        -0.54021994, -0.40366273, -0.32146206, -0.26967639, -0.22249178,
        -0.1743709 , -0.11035225, -0.03037839,  0.05145825,  0.14062265,
         0.23436737,  0.37227278,  0.59693567,  0.83389014,  1.03226944,
         1.71732196],
       [-1. , -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0. ,
         0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ])
}


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController
  RadarInterface = RadarInterface

  DRIVABLE_GEARS = (structs.CarState.GearShifter.sport, structs.CarState.GearShifter.low,
                    structs.CarState.GearShifter.eco, structs.CarState.GearShifter.manumatic)

  @staticmethod
  def get_pid_accel_limits(CP, current_speed, cruise_speed):
    return CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX

  # Determined by iteratively plotting and minimizing error for f(angle, speed) = steer.
  @staticmethod
  def get_steer_feedforward_volt(desired_angle, v_ego):
    desired_angle *= 0.02904609
    sigmoid = desired_angle / (1 + fabs(desired_angle))
    return 0.10006696 * sigmoid * (v_ego + 3.12485927)

  def get_steer_feedforward_function(self):
    if self.CP.carFingerprint == CAR.CHEVROLET_VOLT:
      return self.get_steer_feedforward_volt
    else:
      return CarInterfaceBase.get_steer_feedforward_default

  def torque_from_lateral_accel(self) -> TorqueFromLateralAccelCallbackType:
    if self.CP.carFingerprint in PL_TORQUE_PARAMS:
      lataccel_values, torque_values = PL_TORQUE_PARAMS[self.CP.carFingerprint]

      def torque_from_lateral_accel_pl(lateral_acceleration: float, torque_params: structs.CarParams.LateralTorqueTuning):
        return np.interp(lateral_acceleration, lataccel_values, torque_values)
      return torque_from_lateral_accel_pl
    else:
      return self.torque_from_lateral_accel_linear

  def lateral_accel_from_torque(self) -> LateralAccelFromTorqueCallbackType:
    if self.CP.carFingerprint in PL_TORQUE_PARAMS:
      torque_values, lataccel_values = PL_TORQUE_PARAMS[self.CP.carFingerprint]

      def lateral_accel_from_torque_pl(torque: float, torque_params: structs.CarParams.LateralTorqueTuning):
        return np.interp(torque, torque_values, lataccel_values)
      return lateral_accel_from_torque_pl
    else:
      return self.lateral_accel_from_torque_linear

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "gm"
    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.gm)]
    ret.autoResumeSng = False
    ret.enableBsm = 0x142 in fingerprint[CanBus.POWERTRAIN]

    if candidate in EV_CAR:
      ret.transmissionType = TransmissionType.direct
      ret.safetyConfigs[0].safetyParam |= GMSafetyFlags.EV.value
    else:
      ret.transmissionType = TransmissionType.automatic

    ret.longitudinalTuning.kiBP = [5., 35.]

    if candidate in (CAMERA_ACC_CAR | SDGM_CAR):
      ret.alphaLongitudinalAvailable = candidate not in SDGM_CAR
      ret.networkLocation = NetworkLocation.fwdCamera
      ret.radarUnavailable = True  # no radar
      ret.pcmCruise = True
      ret.safetyConfigs[0].safetyParam |= GMSafetyFlags.HW_CAM.value
      ret.minEnableSpeed = -1 if candidate in SDGM_CAR else 5 * CV.KPH_TO_MS
      ret.minSteerSpeed = 10 * CV.KPH_TO_MS

      # Tuning for experimental long
      ret.longitudinalTuning.kiV = [2.0, 1.5]
      ret.stoppingDecelRate = 2.0  # reach brake quickly after enabling
      ret.vEgoStopping = 0.25
      ret.vEgoStarting = 0.25

      if alpha_long:
        ret.pcmCruise = False
        ret.openpilotLongitudinalControl = True
        ret.safetyConfigs[0].safetyParam |= GMSafetyFlags.HW_CAM_LONG.value

      if candidate in ALT_ACCS:
        ret.alphaLongitudinalAvailable = False
        ret.openpilotLongitudinalControl = False
        ret.minEnableSpeed = -1.  # engage speed is decided by PCM

    else:  # ASCM, OBD-II harness
      ret.openpilotLongitudinalControl = True
      ret.networkLocation = NetworkLocation.gateway
      # LRR messages can take up to a few seconds to start sending after ignition, check camera data as well which starts earlier
      ret.radarUnavailable = RADAR_HEADER_MSG not in fingerprint[CanBus.OBSTACLE] and CAMERA_DATA_HEADER_MSG not in fingerprint[CanBus.OBSTACLE] and not docs
      ret.pcmCruise = False  # stock non-adaptive cruise control is kept off
      # supports stop and go, but initial engage must (conservatively) be above 18mph
      ret.minEnableSpeed = 18 * CV.MPH_TO_MS
      ret.minSteerSpeed = 7 * CV.MPH_TO_MS

      # Tuning
      ret.longitudinalTuning.kiV = [2.4, 1.5]

    # These cars have been put into dashcam only due to both a lack of users and test coverage.
    # These cars likely still work fine. Once a user confirms each car works and a test route is
    # added to opendbc/car/tests/routes.py, we can remove it from this list.
    ret.dashcamOnly = candidate in {CAR.CADILLAC_ATS, CAR.HOLDEN_ASTRA, CAR.CHEVROLET_MALIBU, CAR.BUICK_REGAL} or \
                      (ret.networkLocation == NetworkLocation.gateway and ret.radarUnavailable)

    # Start with a baseline tuning for all GM vehicles. Override tuning as needed in each model section below.
    ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]
    ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.00]]
    ret.lateralTuning.pid.kf = 0.00004   # full torque for 20 deg at 80mph means 0.00007818594
    ret.steerActuatorDelay = 0.1  # Default delay, not measured yet

    ret.steerLimitTimer = 0.4
    ret.longitudinalActuatorDelay = 0.5  # large delay to initially start braking

    if candidate == CAR.CHEVROLET_VOLT:
      ret.lateralTuning.pid.kpBP = [0., 40.]
      ret.lateralTuning.pid.kpV = [0., 0.17]
      ret.lateralTuning.pid.kiBP = [0.]
      ret.lateralTuning.pid.kiV = [0.]
      ret.lateralTuning.pid.kf = 1.  # get_steer_feedforward_volt()
      ret.steerActuatorDelay = 0.2

    elif candidate == CAR.GMC_ACADIA:
      ret.minEnableSpeed = -1.  # engage speed is decided by pcm
      ret.steerActuatorDelay = 0.2
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate == CAR.BUICK_LACROSSE:
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate == CAR.CADILLAC_ESCALADE:
      ret.minEnableSpeed = -1.  # engage speed is decided by pcm
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate in (CAR.CADILLAC_ESCALADE_ESV, CAR.CADILLAC_ESCALADE_ESV_2019):
      ret.minEnableSpeed = -1.  # engage speed is decided by pcm

      if candidate == CAR.CADILLAC_ESCALADE_ESV:
        ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[10., 41.0], [10., 41.0]]
        ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.13, 0.24], [0.01, 0.02]]
        ret.lateralTuning.pid.kf = 0.000045
      else:
        ret.steerActuatorDelay = 0.2
        CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate == CAR.CHEVROLET_BOLT_EUV:
      ret.steerActuatorDelay = 0.2
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate == CAR.CHEVROLET_SILVERADO:
      # On the Bolt, the ECM and camera independently check that you are either above 5 kph or at a stop
      # with foot on brake to allow engagement, but this platform only has that check in the camera.
      # TODO: check if this is split by EV/ICE with more platforms in the future
      if ret.openpilotLongitudinalControl:
        ret.minEnableSpeed = -1.
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate == CAR.CHEVROLET_EQUINOX:
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate == CAR.CHEVROLET_TRAILBLAZER:
      ret.steerActuatorDelay = 0.2
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate == CAR.CADILLAC_XT4:
      ret.steerActuatorDelay = 0.2
      ret.minSteerSpeed = 30 * CV.MPH_TO_MS
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate == CAR.CHEVROLET_VOLT_2019:
      ret.steerActuatorDelay = 0.2
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate == CAR.CHEVROLET_TRAVERSE:
      ret.steerActuatorDelay = 0.2
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate == CAR.GMC_YUKON:
      ret.steerActuatorDelay = 0.5
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)
      ret.dashcamOnly = True  # Needs steerRatio, tireStiffness, and lat accel factor tuning

    return ret
