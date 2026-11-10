/** @file
 *  @brief MAVLink comm protocol generated from throwmav.xml
 *  @see http://mavlink.org
 */
#pragma once
#ifndef MAVLINK_THROWMAV_H
#define MAVLINK_THROWMAV_H

#ifndef MAVLINK_H
    #error Wrong include order: MAVLINK_THROWMAV.H MUST NOT BE DIRECTLY USED. Include mavlink.h from the same directory instead or set ALL AND EVERY defines from MAVLINK.H manually accordingly, including the #define MAVLINK_H call.
#endif

#define MAVLINK_THROWMAV_XML_HASH -4370938528121870477

#ifdef __cplusplus
extern "C" {
#endif

// MESSAGE LENGTHS AND CRCS

#ifndef MAVLINK_MESSAGE_LENGTHS
#define MAVLINK_MESSAGE_LENGTHS {}
#endif

#ifndef MAVLINK_MESSAGE_CRCS
#define MAVLINK_MESSAGE_CRCS {{0, 50, 9, 9, 0, 0, 0}, {1, 124, 31, 43, 0, 0, 0}, {2, 137, 12, 12, 0, 0, 0}, {11, 89, 6, 6, 1, 4, 0}, {20, 214, 20, 20, 3, 2, 3}, {22, 220, 25, 25, 0, 0, 0}, {23, 168, 23, 23, 3, 4, 5}, {24, 24, 30, 52, 0, 0, 0}, {27, 144, 26, 29, 0, 0, 0}, {31, 246, 32, 48, 0, 0, 0}, {32, 185, 28, 28, 0, 0, 0}, {65, 118, 42, 42, 0, 0, 0}, {76, 152, 33, 33, 3, 30, 31}, {77, 143, 3, 10, 3, 8, 9}, {82, 49, 39, 39, 3, 36, 37}, {84, 143, 53, 53, 3, 50, 51}, {193, 71, 22, 26, 0, 0, 0}, {245, 130, 2, 2, 0, 0, 0}, {253, 83, 51, 54, 0, 0, 0}, {258, 187, 32, 232, 3, 0, 1}}
#endif

#include "../protocol.h"

#define MAVLINK_ENABLED_THROWMAV

// ENUM DEFINITIONS



// MAVLINK VERSION

#ifndef MAVLINK_VERSION
#define MAVLINK_VERSION 3
#endif

#if (MAVLINK_VERSION == 0)
#undef MAVLINK_VERSION
#define MAVLINK_VERSION 3
#endif

// MESSAGE DEFINITIONS
#include "./mavlink_msg_heartbeat.h"
#include "./mavlink_msg_sys_status.h"
#include "./mavlink_msg_system_time.h"
#include "./mavlink_msg_set_mode.h"
#include "./mavlink_msg_param_request_read.h"
#include "./mavlink_msg_param_value.h"
#include "./mavlink_msg_param_set.h"
#include "./mavlink_msg_gps_raw_int.h"
#include "./mavlink_msg_raw_imu.h"
#include "./mavlink_msg_attitude_quaternion.h"
#include "./mavlink_msg_local_position_ned.h"
#include "./mavlink_msg_rc_channels.h"
#include "./mavlink_msg_command_long.h"
#include "./mavlink_msg_command_ack.h"
#include "./mavlink_msg_set_attitude_target.h"
#include "./mavlink_msg_set_position_target_local_ned.h"
#include "./mavlink_msg_ekf_status_report.h"
#include "./mavlink_msg_extended_sys_state.h"
#include "./mavlink_msg_statustext.h"
#include "./mavlink_msg_play_tune.h"

// base include



#if MAVLINK_THROWMAV_XML_HASH == MAVLINK_PRIMARY_XML_HASH
# define MAVLINK_MESSAGE_INFO {MAVLINK_MESSAGE_INFO_HEARTBEAT, MAVLINK_MESSAGE_INFO_SYS_STATUS, MAVLINK_MESSAGE_INFO_SYSTEM_TIME, MAVLINK_MESSAGE_INFO_SET_MODE, MAVLINK_MESSAGE_INFO_PARAM_REQUEST_READ, MAVLINK_MESSAGE_INFO_PARAM_VALUE, MAVLINK_MESSAGE_INFO_PARAM_SET, MAVLINK_MESSAGE_INFO_GPS_RAW_INT, MAVLINK_MESSAGE_INFO_RAW_IMU, MAVLINK_MESSAGE_INFO_ATTITUDE_QUATERNION, MAVLINK_MESSAGE_INFO_LOCAL_POSITION_NED, MAVLINK_MESSAGE_INFO_RC_CHANNELS, MAVLINK_MESSAGE_INFO_COMMAND_LONG, MAVLINK_MESSAGE_INFO_COMMAND_ACK, MAVLINK_MESSAGE_INFO_SET_ATTITUDE_TARGET, MAVLINK_MESSAGE_INFO_SET_POSITION_TARGET_LOCAL_NED, MAVLINK_MESSAGE_INFO_EKF_STATUS_REPORT, MAVLINK_MESSAGE_INFO_EXTENDED_SYS_STATE, MAVLINK_MESSAGE_INFO_STATUSTEXT, MAVLINK_MESSAGE_INFO_PLAY_TUNE}
# define MAVLINK_MESSAGE_NAMES {{ "ATTITUDE_QUATERNION", 31 }, { "COMMAND_ACK", 77 }, { "COMMAND_LONG", 76 }, { "EKF_STATUS_REPORT", 193 }, { "EXTENDED_SYS_STATE", 245 }, { "GPS_RAW_INT", 24 }, { "HEARTBEAT", 0 }, { "LOCAL_POSITION_NED", 32 }, { "PARAM_REQUEST_READ", 20 }, { "PARAM_SET", 23 }, { "PARAM_VALUE", 22 }, { "PLAY_TUNE", 258 }, { "RAW_IMU", 27 }, { "RC_CHANNELS", 65 }, { "SET_ATTITUDE_TARGET", 82 }, { "SET_MODE", 11 }, { "SET_POSITION_TARGET_LOCAL_NED", 84 }, { "STATUSTEXT", 253 }, { "SYSTEM_TIME", 2 }, { "SYS_STATUS", 1 }}
# if MAVLINK_COMMAND_24BIT
#  include "../mavlink_get_info.h"
# endif
#endif

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // MAVLINK_THROWMAV_H
