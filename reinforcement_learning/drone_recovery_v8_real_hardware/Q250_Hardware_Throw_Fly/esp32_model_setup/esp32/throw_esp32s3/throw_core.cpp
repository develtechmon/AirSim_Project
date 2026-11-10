/* throw_core.cpp - REAL THROW on the ESP32-S3: port of throw_real.py (version d, 8 Nov 2026).
   Every rule, number, printout and file column is the same as throw_real.py / impact_check.py
   (section numbers in comments refer to the throw_real.py header). Differences (ESP32 only):
   - the network is plain C (ppo_mlp.cpp, int16 weights, < 0.0001 from the ONNX), not onnxruntime
   - steps are kept in RAM during the catch and written to flash during the GUIDED hold / LAND
     (flash writes pause the CPU for up to ~50 ms); READY keeps the last 1 s before the release
   - files are named by power-up number (run<boot>...), the date comes from the GPS (SYSTEM_TIME) */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "throw_core.h"
#include "ppo_mlp.h"
#include "ppo_weights.h"
#include "src/mavlink/throwmav/mavlink.h"

/* MAVLink numbers used (MAVLink common.xml / minimal.xml; the small library has no enum tables) */
static const int MAV_TYPE_GCS = 6, MAV_AUTOPILOT_INVALID = 8, MAV_MODE_FLAG_SAFETY_ARMED = 128;
static const int MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1, MAV_PARAM_TYPE_REAL32 = 9, MAV_FRAME_LOCAL_NED = 1;
static const uint16_t MAV_CMD_DO_SET_MODE = 176, MAV_CMD_COMPONENT_ARM_DISARM = 400, MAV_CMD_SET_MESSAGE_INTERVAL = 511;

/* ---------------------------------------------------------------- settings (= throw_real.py) */
static const int THROW_CH = 5;
static const float HOLD_S = 10.0f, MAX_UP_M = 5.0f, HZ = 50.0f;
static const float FREEFALL_G = 0.35f, PUSH_G = 1.5f, PUSH_TO_FREE_S = 0.8f;
static const int FREEFALL_N = 3;
static const float SPOOL_MAX_S = 1.0f, RECOVER_MAX_S = 3.0f, STALE_S = 0.5f, HOLD_K = 0.4f;
static const float VZ_OK = 1.0f, STOP_V = 0.2f, LAND_WAIT_S = 60.0f;
static const int CH_HIGH = 1700, CH_LOW = 1300, CH6_LO = 1300, CH6_HI = 1700, CH3_LOW = 1300;
static const int MIN_SATS = 5;
static const float MIN_VOLT = 14.9f;
static const int LANDED_ON_GROUND = 1;
/* impact_check.py */
static const float HELD_S = 0.5f, RATE_OK = 1.2f, TILT_OK = 30.0f, HSPEED_OK = 3.0f;
static const float BC_MAX_LEAN = 0.6f, YAW_BRAKE_T = 0.5f, YAW_BRAKE_MAX = 0.8f, YAW_BRAKE_UP = 0.7f;
static const float RULE_SIDE_K = 0.15f, RULE_SIDE_MAX = 0.45f, RULE_UP_MIN = 0.3f, RULE_THR_INV = 1.0f;
static const float RULE_BRAKE = 0.25f;
static const int GUID_THRUST_BIT = 8;
static const float JUMP_M = 0.5f;                                   /* ap_link.py */
static const char *TUNE_READY = "MFT240L8O6cP8cP8c";               /* 3 short beeps */
static const char *TUNE_REFUSED = "MFT100L2O4cP4c";                /* 2 long beeps */
static const char *TUNE_START_RULE = "MFT120L2O5cP8L8c";           /* 1 long + 1 short */
static const char *TUNE_START_PPO = "MFT120L2O5cP8L8cP8c";         /* 1 long + 2 short */
static const char *LIMIT_NAMES[2][3] = {{"ATC_ACCEL_R_MAX", "ATC_ACCEL_P_MAX", "ATC_ACCEL_Y_MAX"},
                                        {"ATC_ACC_R_MAX", "ATC_ACC_P_MAX", "ATC_ACC_Y_MAX"}};
static const char *REAL_LIMIT_NAMES[3] = {"ATC_ACCEL_R_MAX", "ATC_ACCEL_P_MAX", "ATC_ACCEL_Y_MAX"};
static const float REAL_LIMIT_VALUES[3] = {282459.0f, 282459.0f, 31500.0f};

/* ---------------------------------------------------------------- small helpers */
static Hal *H;
static char LINE[320];
static void P(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(LINE, sizeof(LINE), fmt, ap);
    va_end(ap);
    H->out(LINE);
}
static float now_s() { return H->ms() / 1000.0f; }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ---------------------------------------------------------------- the MAVLink link (= ap_link.py) */
enum { M_IMU, M_ESS, M_RC, M_SYS, M_GPS, M_EKF, M_TIME, M_N };
struct Link {
    uint8_t tsys = 0, tcomp = 1;
    bool have_hb = false, armed = false;
    char mode[16] = "?";
    bool have_q = false, have_pos = false;
    float q[4] = {1, 0, 0, 0}, rates[3] = {0, 0, 0}, pos[3] = {0, 0, 0}, vel[3] = {0, 0, 0};
    float last_raw[3] = {0, 0, 0}, off[2] = {0, 0};
    bool have_raw = false;
    uint32_t n_att = 0, n_imu = 0;
    bool has[M_N] = {false};
    uint32_t t_ms[M_N] = {0};
    mavlink_raw_imu_t imu;
    mavlink_extended_sys_state_t ess;
    mavlink_rc_channels_t rc;
    mavlink_sys_status_t sys;
    mavlink_gps_raw_int_t gps;
    mavlink_ekf_status_report_t ekf;
    mavlink_system_time_t st;
};
static Link L;

struct Param { char name[17]; float v; bool got; };
static Param PARAMS[12];
static int N_PARAMS = 0;
static Param *param_slot(const char *name)
{
    for (int i = 0; i < N_PARAMS; i++)
        if (!strcmp(PARAMS[i].name, name)) return &PARAMS[i];
    if (N_PARAMS >= 12) return nullptr;
    Param *p = &PARAMS[N_PARAMS++];
    strncpy(p->name, name, 16);
    p->name[16] = 0;
    p->got = false;
    p->v = 0;
    return p;
}
static bool param_get(const char *name, float &v)
{
    for (int i = 0; i < N_PARAMS; i++)
        if (!strcmp(PARAMS[i].name, name) && PARAMS[i].got) { v = PARAMS[i].v; return true; }
    return false;
}

static const char *mode_name(uint32_t m)
{
    switch (m) {
    case 0: return "STABILIZE"; case 1: return "ACRO"; case 2: return "ALT_HOLD"; case 3: return "AUTO";
    case 4: return "GUIDED"; case 5: return "LOITER"; case 6: return "RTL"; case 7: return "CIRCLE";
    case 9: return "LAND"; case 11: return "DRIFT"; case 13: return "SPORT"; case 14: return "FLIP";
    case 15: return "AUTOTUNE"; case 16: return "POSHOLD"; case 17: return "BRAKE"; case 18: return "THROW";
    case 19: return "AVOID_ADSB"; case 20: return "GUIDED_NOGPS"; case 21: return "SMART_RTL";
    case 22: return "FLOWHOLD"; case 23: return "FOLLOW"; case 24: return "ZIGZAG"; case 25: return "SYSTEMID";
    case 26: return "AUTOROTATE"; case 27: return "AUTO_RTL"; case 28: return "TURTLE";
    default: return "?";
    }
}
static int mode_number(const char *name)
{
    for (int m = 0; m < 30; m++)
        if (!strcmp(mode_name(m), name)) return m;
    return -1;
}

static void handle(const mavlink_message_t &msg)
{
    if (msg.msgid == MAVLINK_MSG_ID_HEARTBEAT) {
        mavlink_heartbeat_t hb;
        mavlink_msg_heartbeat_decode(&msg, &hb);
        if (hb.type == MAV_TYPE_GCS || hb.autopilot == MAV_AUTOPILOT_INVALID) return;
        if (!L.have_hb) { L.tsys = msg.sysid; L.tcomp = msg.compid; L.have_hb = true; }
        if (msg.sysid != L.tsys || msg.compid != L.tcomp) return;
        L.armed = (hb.base_mode & MAV_MODE_FLAG_SAFETY_ARMED) != 0;
        strncpy(L.mode, mode_name(hb.custom_mode), sizeof(L.mode) - 1);
        return;
    }
    if (!L.have_hb || msg.sysid != L.tsys) return;
    uint32_t t = H->ms();
    switch (msg.msgid) {
    case MAVLINK_MSG_ID_LOCAL_POSITION_NED: {
        mavlink_local_position_ned_t p;
        mavlink_msg_local_position_ned_decode(&msg, &p);
        float raw[3] = {p.x, p.y, p.z};
        if (L.have_raw) {
            float jx = raw[0] - L.last_raw[0], jy = raw[1] - L.last_raw[1];
            if (sqrtf(jx * jx + jy * jy) > JUMP_M) {                 /* EKF reset, not real motion */
                L.off[0] -= jx;
                L.off[1] -= jy;
                P("[link] EKF position reset of %.2f m removed (position kept smooth)", sqrtf(jx * jx + jy * jy));
            }
        }
        memcpy(L.last_raw, raw, sizeof(raw));
        L.have_raw = true;
        L.pos[0] = raw[0] + L.off[0];
        L.pos[1] = raw[1] + L.off[1];
        L.pos[2] = raw[2];
        L.vel[0] = p.vx; L.vel[1] = p.vy; L.vel[2] = p.vz;
        L.have_pos = true;
        break;
    }
    case MAVLINK_MSG_ID_ATTITUDE_QUATERNION: {
        mavlink_attitude_quaternion_t a;
        mavlink_msg_attitude_quaternion_decode(&msg, &a);
        L.q[0] = a.q1; L.q[1] = a.q2; L.q[2] = a.q3; L.q[3] = a.q4;
        L.rates[0] = a.rollspeed; L.rates[1] = a.pitchspeed; L.rates[2] = a.yawspeed;
        L.have_q = true;
        L.n_att++;
        break;
    }
    case MAVLINK_MSG_ID_RAW_IMU: mavlink_msg_raw_imu_decode(&msg, &L.imu); L.has[M_IMU] = true; L.t_ms[M_IMU] = t; L.n_imu++; break;
    case MAVLINK_MSG_ID_EXTENDED_SYS_STATE: mavlink_msg_extended_sys_state_decode(&msg, &L.ess); L.has[M_ESS] = true; L.t_ms[M_ESS] = t; break;
    case MAVLINK_MSG_ID_RC_CHANNELS: mavlink_msg_rc_channels_decode(&msg, &L.rc); L.has[M_RC] = true; L.t_ms[M_RC] = t; break;
    case MAVLINK_MSG_ID_SYS_STATUS: mavlink_msg_sys_status_decode(&msg, &L.sys); L.has[M_SYS] = true; L.t_ms[M_SYS] = t; break;
    case MAVLINK_MSG_ID_GPS_RAW_INT: mavlink_msg_gps_raw_int_decode(&msg, &L.gps); L.has[M_GPS] = true; L.t_ms[M_GPS] = t; break;
    case MAVLINK_MSG_ID_EKF_STATUS_REPORT: mavlink_msg_ekf_status_report_decode(&msg, &L.ekf); L.has[M_EKF] = true; L.t_ms[M_EKF] = t; break;
    case MAVLINK_MSG_ID_SYSTEM_TIME: mavlink_msg_system_time_decode(&msg, &L.st); L.has[M_TIME] = true; L.t_ms[M_TIME] = t; break;
    case MAVLINK_MSG_ID_STATUSTEXT: {
        mavlink_statustext_t s;
        mavlink_msg_statustext_decode(&msg, &s);
        char txt[51];
        memcpy(txt, s.text, 50);
        txt[50] = 0;
        P("[fc] %s", txt);
        break;
    }
    case MAVLINK_MSG_ID_PARAM_VALUE: {
        mavlink_param_value_t pv;
        mavlink_msg_param_value_decode(&msg, &pv);
        char id[17];
        memcpy(id, pv.param_id, 16);
        id[16] = 0;
        for (int i = 0; i < N_PARAMS; i++)
            if (!strcmp(PARAMS[i].name, id)) { PARAMS[i].v = pv.param_value; PARAMS[i].got = true; }
        break;
    }
    default: break;
    }
}

static void update()          /* drain all waiting bytes (= APLink.update) */
{
    static mavlink_message_t msg;
    static mavlink_status_t st;
    uint8_t buf[256];
    for (;;) {
        int n = H->fc_read(buf, sizeof(buf));
        if (n <= 0) return;
        for (int i = 0; i < n; i++)
            if (mavlink_parse_char(MAVLINK_COMM_0, buf[i], &msg, &st)) handle(msg);
    }
}

static void wait_ms(uint32_t ms)  /* time.sleep(), but keep reading the FC */
{
    uint32_t t_end = H->ms() + ms;
    while ((int32_t)(H->ms() - t_end) < 0) {
        update();
        H->sleep_ms(1);
    }
    update();
}

static uint32_t step_next = 0;    /* RateKeeper(50 Hz) */
static void rk_start() { step_next = H->ms(); }
static void rk_wait()
{
    step_next += (uint32_t)(1000.0f / HZ);
    int32_t late = (int32_t)(H->ms() - step_next);
    if (late > 100) step_next = H->ms();                         /* far behind: do not catch up */
    while ((int32_t)(H->ms() - step_next) < 0) {
        update();
        if ((int32_t)(step_next - H->ms()) > 2) H->sleep_ms(1);
    }
}

static uint8_t TXBUF[MAVLINK_MAX_PACKET_LEN];
static void send(mavlink_message_t &m)
{
    uint16_t n = mavlink_msg_to_send_buffer(TXBUF, &m);
    H->fc_write(TXBUF, n);
}
static const uint8_t MY_SYS = 255, MY_COMP = 0;                 /* same as pymavlink (source_system=255) */
static void command_long(uint16_t cmd, float p1, float p2 = 0, float p3 = 0, float p4 = 0, float p5 = 0, float p6 = 0, float p7 = 0)
{
    mavlink_message_t m;
    mavlink_msg_command_long_pack(MY_SYS, MY_COMP, &m, L.tsys, L.tcomp, cmd, 0, p1, p2, p3, p4, p5, p6, p7);
    send(m);
}
static void request_rate(uint32_t msg_id, float hz) { command_long(MAV_CMD_SET_MESSAGE_INTERVAL, (float)msg_id, 1e6f / hz); }
static void set_mode(const char *name)
{
    int m = mode_number(name);
    if (m >= 0) command_long(MAV_CMD_DO_SET_MODE, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, (float)m);
}
static void send_params1(const char *name, float v)
{
    mavlink_message_t m;
    char id[17] = {0};
    strncpy(id, name, 16);
    mavlink_msg_param_set_pack(MY_SYS, MY_COMP, &m, L.tsys, L.tcomp, id, v, MAV_PARAM_TYPE_REAL32);
    send(m);
}
static void get_params(const char *const *names, int n, float timeout_s = 5.0f)
{
    for (int i = 0; i < n; i++) {
        Param *p = param_slot(names[i]);
        if (p) p->got = false;
        mavlink_message_t m;
        char id[17] = {0};
        strncpy(id, names[i], 16);
        mavlink_msg_param_request_read_pack(MY_SYS, MY_COMP, &m, L.tsys, L.tcomp, id, -1);
        send(m);
    }
    uint32_t t_end = H->ms() + (uint32_t)(timeout_s * 1000);
    while ((int32_t)(H->ms() - t_end) < 0) {
        update();
        int got = 0;
        float v;
        for (int i = 0; i < n; i++) got += param_get(names[i], v);
        if (got == n) return;
        H->sleep_ms(2);
    }
}
static void beep(const char *tune)
{
    mavlink_message_t m;
    char t1[30] = {0}, t2[200] = {0};
    strncpy(t1, tune, 29);
    mavlink_msg_play_tune_pack(MY_SYS, MY_COMP, &m, L.tsys, L.tcomp, t1, t2);
    send(m);
}
static void send_attitude(const float q[4], float thrust)    /* impact_check.send_attitude: type_mask 7 */
{
    mavlink_message_t m;
    float qq[4] = {q[0], q[1], q[2], q[3]};
    mavlink_msg_set_attitude_target_pack(MY_SYS, MY_COMP, &m, 0, L.tsys, L.tcomp, 7, qq, 0, 0, 0, thrust);
    send(m);
}
static void send_velocity(float vx, float vy, float vz)        /* ap_link.send_velocity: VEL_MASK, LOCAL_NED */
{
    mavlink_message_t m;
    const uint16_t VEL_MASK = 1 | 2 | 4 | 64 | 128 | 256 | 1024;
    mavlink_msg_set_position_target_local_ned_pack(MY_SYS, MY_COMP, &m, 0, L.tsys, L.tcomp, MAV_FRAME_LOCAL_NED,
                                                   VEL_MASK, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0);
    send(m);
}
static bool request_mode(const char *mode, float timeout_s = 1.0f)
{
    uint32_t t_end = H->ms() + (uint32_t)(timeout_s * 1000);
    while ((int32_t)(H->ms() - t_end) < 0) {
        set_mode(mode);
        wait_ms(100);
        if (!strcmp(L.mode, mode)) return true;
    }
    return false;
}
static void disarm()
{
    command_long(MAV_CMD_COMPONENT_ARM_DISARM, 0);
    uint32_t t_end = H->ms() + 2000;
    while ((int32_t)(H->ms() - t_end) < 0 && L.armed) wait_ms(50);
}

/* ---------------------------------------------------------------- readings (= throw_real.py helpers) */
static bool accel_g(float &g)
{
    if (!L.has[M_IMU]) return false;
    float x = L.imu.xacc, y = L.imu.yacc, z = L.imu.zacc;
    g = sqrtf(x * x + y * y + z * z) / 1000.0f;
    return true;
}
static int landed_state() { return L.has[M_ESS] ? L.ess.landed_state : 0; }
static int rc(int ch)
{
    if (!L.has[M_RC]) return 0;
    const mavlink_rc_channels_t &c = L.rc;
    const uint16_t v[10] = {0, c.chan1_raw, c.chan2_raw, c.chan3_raw, c.chan4_raw, c.chan5_raw, c.chan6_raw,
                            c.chan7_raw, c.chan8_raw, c.chan9_raw};
    return (ch >= 1 && ch <= 9) ? v[ch] : 0;
}
static float volts() { return (L.has[M_SYS] && L.sys.voltage_battery != 65535) ? L.sys.voltage_battery / 1000.0f : 0.0f; }
static void gps(int &fix, int &sats)
{
    fix = L.has[M_GPS] ? L.gps.fix_type : 0;
    sats = L.has[M_GPS] ? L.gps.satellites_visible : 0;
}
static bool ekf_gps() { return L.has[M_EKF] && (L.ekf.flags & 16); }
static float alt() { return -L.pos[2]; }
static float tilt_deg()
{
    float up = 1.0f - 2.0f * (L.q[1] * L.q[1] + L.q[2] * L.q[2]);
    return acosf(clampf(up, -1.0f, 1.0f)) * 57.2957795f;
}
static float rate_norm() { return sqrtf(L.rates[0] * L.rates[0] + L.rates[1] * L.rates[1] + L.rates[2] * L.rates[2]); }
static float hspeed() { return sqrtf(L.vel[0] * L.vel[0] + L.vel[1] * L.vel[1]); }

/* ---------------------------------------------------------------- maths (= impact_check.py) */
static void quat_to_R(const float q[4], float R[3][3])
{
    float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0][0] = 1 - 2 * (y * y + z * z); R[0][1] = 2 * (x * y - w * z);     R[0][2] = 2 * (x * z + w * y);
    R[1][0] = 2 * (x * y + w * z);     R[1][1] = 1 - 2 * (x * x + z * z); R[1][2] = 2 * (y * z - w * x);
    R[2][0] = 2 * (x * z - w * y);     R[2][1] = 2 * (y * z + w * x);     R[2][2] = 1 - 2 * (x * x + y * y);
}
static void euler_quat(float roll, float pitch, float yaw, float q[4])
{
    float cr = cosf(roll / 2), sr = sinf(roll / 2), cp = cosf(pitch / 2), sp = sinf(pitch / 2);
    float cy = cosf(yaw / 2), sy = sinf(yaw / 2);
    q[0] = cr * cp * cy + sr * sp * sy;
    q[1] = sr * cp * cy - cr * sp * sy;
    q[2] = cr * sp * cy + sr * cp * sy;
    q[3] = cr * cp * sy - sr * sp * cy;
}
static void reframe(float roll, float pitch, float yaw_from, float yaw_to, float &r_out, float &p_out)
{
    if (yaw_to == yaw_from) { r_out = roll; p_out = pitch; return; }
    float fx = tanf(pitch), fy = -tanf(roll);
    float c = cosf(yaw_to - yaw_from), s = sinf(yaw_to - yaw_from);
    r_out = -atanf(-s * fx + c * fy);
    p_out = atanf(c * fx + s * fy);
}
static void heading_target(float R[3][3], float &yaw_now, float &yaw_tgt)  /* YAW BRAKE */
{
    yaw_now = atan2f(R[1][0], R[0][0]);
    yaw_tgt = yaw_now;
    if (YAW_BRAKE_T > 0 && R[2][2] > YAW_BRAKE_UP)
        yaw_tgt = yaw_now - clampf(YAW_BRAKE_T * L.rates[2], -YAW_BRAKE_MAX, YAW_BRAKE_MAX);
}
static void level_q(float q[4])
{
    float R[3][3];
    quat_to_R(L.q, R);
    euler_quat(0, 0, atan2f(R[1][0], R[0][0]), q);
}
static void height_hold(float hover, float q[4], float &thr)
{
    float up = 1.0f - 2.0f * (L.q[1] * L.q[1] + L.q[2] * L.q[2]);
    if (up < 0.5f) up = 0.5f;
    thr = clampf(hover * (1.0f + HOLD_K * L.vel[2]) / up, 0.3f * hover, 1.0f);
    level_q(q);
}
static void attitude_rule(float hover, float q[4], float &thr)       /* rule_targets + attitude_rule */
{
    float R[3][3];
    quat_to_R(L.q, R);
    float up = R[2][2], td[3] = {0, 0, 1};
    if (up > 0.7f) {
        float lx = clampf(RULE_SIDE_K * L.vel[0], -RULE_SIDE_MAX, RULE_SIDE_MAX);
        float ly = clampf(RULE_SIDE_K * L.vel[1], -RULE_SIDE_MAX, RULE_SIDE_MAX);
        float n = sqrtf(1 + lx * lx + ly * ly);
        td[0] = lx / n; td[1] = ly / n; td[2] = 1 / n;
    }
    float v_down = L.vel[2] > 0 ? L.vel[2] : 0;
    if (up < RULE_UP_MIN) thr = RULE_THR_INV * hover;
    else thr = clampf(hover * (1 + RULE_BRAKE * v_down) / up, RULE_THR_INV * hover, 1.0f);
    float yaw_now, yaw;
    heading_target(R, yaw_now, yaw);
    float cy = cosf(yaw), sy = sinf(yaw);
    float lx = td[0] / td[2], ly = td[1] / td[2];
    float fx = cy * lx + sy * ly, fy = -sy * lx + cy * ly;
    euler_quat(-atanf(fy), atanf(fx), yaw, q);
}
static void bc_observation(float alt0, const float last[3], float obs[13])
{
    float R[3][3];
    quat_to_R(L.q, R);
    float yaw = atan2f(R[1][0], R[0][0]);
    obs[0] = R[2][0]; obs[1] = R[2][1]; obs[2] = R[2][2];
    obs[3] = L.rates[0]; obs[4] = L.rates[1]; obs[5] = L.rates[2];
    obs[6] = L.vel[2];
    obs[7] = alt() - alt0;
    obs[8] = cosf(yaw) * L.vel[0] + sinf(yaw) * L.vel[1];
    obs[9] = -sinf(yaw) * L.vel[0] + cosf(yaw) * L.vel[1];
    obs[10] = last[0]; obs[11] = last[1]; obs[12] = last[2];
}
static void action_to_attitude(const float a[3], float hover, float q[4], float &thr)
{
    float roll = clampf(a[0], -BC_MAX_LEAN, BC_MAX_LEAN), pitch = clampf(a[1], -BC_MAX_LEAN, BC_MAX_LEAN);
    float thr_x = clampf(a[2], 0.0f, 1.0f / hover);
    float R[3][3];
    quat_to_R(L.q, R);
    float yaw_now, yaw, r_cmd, p_cmd;
    heading_target(R, yaw_now, yaw);
    reframe(roll, pitch, yaw_now, yaw, r_cmd, p_cmd);
    euler_quat(r_cmd, p_cmd, yaw, q);
    thr = thr_x * hover;
}
static void ppo_step(float hover, float alt0, float last[3], float q[4], float &thr)  /* = throw_real.ppo_step */
{
    float obs[13], a[3];
    bc_observation(alt0, last, obs);
    ppo_act(obs, a);
    action_to_attitude(a, hover, q, thr);
    last[0] = a[0]; last[1] = a[1]; last[2] = a[2];
}

/* ---------------------------------------------------------------- data requests (version d, change 2) */
static const uint32_t DATA_ID[3] = {MAVLINK_MSG_ID_GPS_RAW_INT, MAVLINK_MSG_ID_SYS_STATUS, MAVLINK_MSG_ID_EKF_STATUS_REPORT};
static const int DATA_IDX[3] = {M_GPS, M_SYS, M_EKF};
static void ask_data_again(float max_age = 3.0f)
{
    uint32_t t = H->ms();
    for (int i = 0; i < 3; i++)
        if (!L.has[DATA_IDX[i]] || t - L.t_ms[DATA_IDX[i]] > (uint32_t)(max_age * 1000)) request_rate(DATA_ID[i], 2);
}

static bool back_to_alt_hold()   /* version d, change 1 */
{
    update();
    if (strcmp(L.mode, "GUIDED_NOGPS") && strcmp(L.mode, "GUIDED") && strcmp(L.mode, "LAND")) return false;
    uint32_t t_end = H->ms() + 3000;
    while (L.armed && (int32_t)(H->ms() - t_end) < 0) wait_ms(50);
    if (L.armed) return false;
    if (strcmp(L.mode, "GUIDED_NOGPS") && strcmp(L.mode, "GUIDED") && strcmp(L.mode, "LAND")) return false;
    if (request_mode("ALT_HOLD", 3)) P("mode back to ALT_HOLD - arm again with CH9 when ready");
    else P("mode stays %s: flick CH8 to another mode and back to ALT_HOLD before arming", L.mode);
    return true;
}

/* ---------------------------------------------------------------- the checks (section 4) */
struct Check { const char *name; bool ok; char txt[64]; };
static Check CHECKS[9];
static bool checks()
{
    int fix, sats;
    gps(fix, sats);
    bool on_ground = landed_state() == LANDED_ON_GROUND;
    float spin_arm = 1, spool = 0.5, rc5 = -1;
    bool has_spin = param_get("MOT_SPIN_ARM", spin_arm), has_spool = param_get("MOT_SPOOL_TIME", spool);
    bool has_rc5 = param_get("RC5_OPTION", rc5);
    int i = 0;
    CHECKS[i].name = "armed by you, on the ground"; CHECKS[i].ok = L.armed && on_ground;
    snprintf(CHECKS[i++].txt, 64, "%s, %s", L.armed ? "armed" : "DISARMED - arm with CH9", on_ground ? "on ground" : "IN THE AIR");
    CHECKS[i].name = "MOT_SPIN_ARM 0"; CHECKS[i].ok = has_spin && fabsf(spin_arm) < 1e-6f;
    snprintf(CHECKS[i++].txt, 64, has_spin ? "%.1f" : "None", spin_arm);
    CHECKS[i].name = "MOT_SPOOL_TIME <= 0.3"; CHECKS[i].ok = has_spool ? spool <= 0.3f + 1e-6f : false;
    snprintf(CHECKS[i++].txt, 64, has_spool ? "%.2f" : "None", spool);
    CHECKS[i].name = "RC5_OPTION 0"; CHECKS[i].ok = has_rc5 && (int)rc5 == 0;
    snprintf(CHECKS[i++].txt, 64, has_rc5 ? "%.1f" : "None", rc5);
    CHECKS[i].name = "CH6 middle (GPS)"; CHECKS[i].ok = rc(6) >= CH6_LO && rc(6) <= CH6_HI;
    snprintf(CHECKS[i++].txt, 64, "CH6 %d", rc(6));
    CHECKS[i].name = "GPS 3D fix, >= 5 sats"; CHECKS[i].ok = fix >= 3 && sats >= MIN_SATS;
    snprintf(CHECKS[i++].txt, 64, "fix %d, %d sats", fix, sats);
    CHECKS[i].name = "EKF using GPS position"; CHECKS[i].ok = ekf_gps();
    snprintf(CHECKS[i++].txt, 64, "%s", ekf_gps() ? "yes" : "NO - wait for 'EKF3 IMU0 is using GPS'");
    CHECKS[i].name = "throttle stick LOW"; CHECKS[i].ok = rc(3) > 900 && rc(3) < CH3_LOW;
    snprintf(CHECKS[i++].txt, 64, "CH3 %d", rc(3));
    CHECKS[i].name = "battery >= 14.9 V"; CHECKS[i].ok = volts() >= MIN_VOLT;
    snprintf(CHECKS[i++].txt, 64, "%.2f V", volts());
    bool all = true;
    for (int k = 0; k < 9; k++) all = all && CHECKS[k].ok;
    return all;
}

/* ---------------------------------------------------------------- outputs (section 6) */
enum Phase { PH_READY, PH_SPOOL, PH_RECOVER, PH_CAUGHT, PH_HOLD, PH_LAND };
static const char *PHASE_NAME[] = {"ready", "spool", "recover", "caught", "hold", "land"};
struct Row {
    float t, tilt, rate, h, hs, dist, g, p, q, r;
    uint16_t n, ch;
    uint8_t phase, landed;
    bool has_g;
};
static const int ROWS_MAX = 600, READY_KEEP = 50;   /* 12 s in RAM; READY keeps its last 1 s */
static Row ROWS[ROWS_MAX];
static int rows_head = 0, rows_count = 0;           /* ring buffer */
static int steps_fh = -1;
static uint32_t rows_dropped = 0;
static bool PPO_MODE = true;
static const char *mode_str() { return PPO_MODE ? "ppo-policy" : "attitude-rule"; }
static char BASE[48];

static void log_row(int n, Phase ph, float t, float h0, const float spot[3], bool has_g, float g)
{
    if (ph == PH_READY && rows_count >= READY_KEEP) { rows_head = (rows_head + 1) % ROWS_MAX; rows_count--; }
    if (rows_count >= ROWS_MAX) { rows_head = (rows_head + 1) % ROWS_MAX; rows_count--; rows_dropped++; }
    Row &r = ROWS[(rows_head + rows_count) % ROWS_MAX];
    rows_count++;
    float dx = L.pos[0] - spot[0], dy = L.pos[1] - spot[1];
    r.n = n; r.phase = ph; r.t = t; r.tilt = tilt_deg(); r.rate = rate_norm(); r.h = alt() - h0;
    r.hs = hspeed(); r.dist = sqrtf(dx * dx + dy * dy); r.has_g = has_g; r.g = g; r.landed = landed_state();
    r.ch = rc(THROW_CH); r.p = L.rates[0]; r.q = L.rates[1]; r.r = L.rates[2];
}
static void flush_rows(int max_rows)    /* write steps to the flash (only when not catching) */
{
    if (steps_fh < 0) return;
    char s[200], g[16];
    for (int k = 0; k < max_rows && rows_count > 0; k++) {
        Row &r = ROWS[rows_head];
        if (r.has_g) snprintf(g, sizeof(g), "%.3f", r.g); else g[0] = 0;
        snprintf(s, sizeof(s), "%u,%s,%s,%.3f,%.1f,%.2f,%.2f,%.2f,%.2f,%s,%u,%u,%.2f,%.2f,%.2f\n", r.n, mode_str(),
                 PHASE_NAME[r.phase], r.t, r.tilt, r.rate, r.h, r.hs, r.dist, g, r.landed, r.ch, r.p, r.q, r.r);
        H->file_write(steps_fh, s);
        rows_head = (rows_head + 1) % ROWS_MAX;
        rows_count--;
    }
}
static void date_str(char *out, int n)
{
    if (L.has[M_TIME] && L.st.time_unix_usec > 1600000000000000ULL) {
        time_t t = (time_t)(L.st.time_unix_usec / 1000000ULL + (H->ms() - L.t_ms[M_TIME]) / 1000);
        struct tm tm;
        gmtime_r(&t, &tm);
        strftime(out, n, "%Y-%m-%dT%H:%M:%SZ", &tm);
    } else {
        snprintf(out, n, "boot+%lus", (unsigned long)(H->ms() / 1000));
    }
}

/* ---------------------------------------------------------------- one throw (section 3) */
struct After {
    bool has = false, has_stop = false, landed_disarmed = false;
    char hold_mode[16] = "", note[64] = "";
    float brake_s = 0, brake_m = 0, wander_m = 0;
};
struct Info {
    int n;
    bool detected, recovered, has_motors, has_upright, has_caught, has_rec, has_low, upright_zero;
    float motors_on_s, upright_s, caught_s, recovered_s, apex, low, peak_rate, drift, volt, hover;
    char ended[64], date[32];
    After after;
};

static After hold_land(int n, float t0_log, float h0, const float spot[3], const char *why)
{
    After a;
    a.has = true;
    bool hold = request_mode("GUIDED");
    if (!hold && !request_mode("LAND")) {
        P("[handover] GUIDED and LAND refused (%s) - mode %s: TAKE OVER (stick middle, CH8 POSHOLD)", why, L.mode);
        strncpy(a.hold_mode, L.mode, sizeof(a.hold_mode) - 1);
        a.has_stop = false;
        return a;
    }
    if (hold) P("[handover] GUIDED hold %d s, then LAND (%s)", (int)HOLD_S, why);
    else P("[handover] no position: LAND now (%s)", why);
    float point[3] = {L.pos[0], L.pos[1], L.pos[2]}, stop_pt[3] = {0, 0, 0};
    bool have_stop = false;
    float t_stop = 0, wander = 0;
    float t0 = now_s();
    rk_start();
    while (L.armed) {
        if (hold && !strcmp(L.mode, "GUIDED")) send_velocity(0, 0, 0);     /* stay here (stick ignored) */
        rk_wait();
        update();
        float g = 0;
        bool hg = accel_g(g);
        log_row(n, !strcmp(L.mode, "GUIDED") ? PH_HOLD : PH_LAND, now_s() - t0_log, h0, spot, hg, g);
        flush_rows(16);                                                   /* write to flash now (not catching) */
        if (!have_stop && hspeed() < STOP_V) {
            memcpy(stop_pt, L.pos, sizeof(stop_pt));
            t_stop = now_s() - t0;
            have_stop = true;
        }
        if (have_stop && !strcmp(L.mode, "GUIDED")) {
            float dx = L.pos[0] - stop_pt[0], dy = L.pos[1] - stop_pt[1];
            float d = sqrtf(dx * dx + dy * dy);
            if (d > wander) wander = d;
        }
        if (!strcmp(L.mode, "GUIDED") && now_s() - t0 >= HOLD_S) {
            request_mode("LAND");
        } else if (strcmp(L.mode, "GUIDED") && strcmp(L.mode, "LAND")) {
            snprintf(a.note, sizeof(a.note), "you took over (%s)", L.mode);
            P("[handover] %s - the script lets go", a.note);
            break;
        }
        if (now_s() - t0 > HOLD_S + LAND_WAIT_S) {
            snprintf(a.note, sizeof(a.note), "LAND did not disarm in 60 s - land it yourself");
            P("[handover] %s", a.note);
            break;
        }
    }
    if (!L.armed) P("[link] landed and disarmed - pick it up now, then CH5 OFF, CH9 OFF");
    strncpy(a.hold_mode, hold ? "GUIDED" : "LAND", sizeof(a.hold_mode) - 1);
    a.has_stop = have_stop;
    if (have_stop) {
        float dx = stop_pt[0] - point[0], dy = stop_pt[1] - point[1];
        a.brake_s = t_stop;
        a.brake_m = sqrtf(dx * dx + dy * dy);
        a.wander_m = wander;
    }
    a.landed_disarmed = !L.armed;
    return a;
}

static float LIMITS_V[3];
static bool HAS_LIMITS = false;
static const char *const *LIMITS_N = nullptr;
static void send_limits(bool zero)
{
    if (!HAS_LIMITS) return;
    for (int i = 0; i < 3; i++) send_params1(LIMITS_N[i], zero ? 0.0f : LIMITS_V[i]);
}

/* returns false if no throw happened (cancel / refused / disarmed while READY) */
static bool one_throw(int n, float hover, int guid, Info &info)
{
    if (!request_mode("GUIDED_NOGPS", 3)) {
        beep(TUNE_REFUSED);
        P("REFUSED by ArduPilot: GUIDED_NOGPS not accepted (mode %s)", L.mode);
        return false;
    }
    update();
    float spot[3] = {L.pos[0], L.pos[1], L.pos[2]}, h0 = alt();          /* the hand = height 0 */
    beep(TUNE_READY);
    P("READY - THROW NOW (3 short beeps, props still, stick stays LOW). CH%d OFF = cancel", THROW_CH);
    float t_ready = now_s();
    Phase phase = PH_READY;
    bool has_push = false, has_rel = false, has_motors = false, has_upright = false, has_caught = false;
    bool has_rec = false, has_held = false, tilted = false, limits_off = false, has_low = false;
    float t_push = 0, t_rel = 0, t_motors = 0, t_upright = 0, t_caught = 0, t_rec = 0, held = 0, spool_t0 = 0;
    float alt_rel = 0, last[3] = {0, 0, 1}, apex = 0, low = 0, peak_rate = 0;
    int n_ff = 0;
    const char *why = "";
    static char why_buf[64];
    float t_att = now_s();
    uint32_t att_seen = L.n_att;
    rows_head = 0; rows_count = 0; rows_dropped = 0;
    rk_start();
    for (;;) {
        float q[4], thr;
        if (phase == PH_READY) { level_q(q); send_attitude(q, 0.5f); }          /* climb-rate mode: stay */
        else if (phase == PH_SPOOL) { level_q(q); send_attitude(q, 1.0f); }     /* take-off = spool up */
        else if (phase == PH_CAUGHT) { height_hold(hover, q, thr); send_attitude(q, thr); }
        else if (!PPO_MODE) { attitude_rule(hover, q, thr); send_attitude(q, thr); }
        else { ppo_step(hover, alt_rel, last, q, thr); send_attitude(q, thr); }
        rk_wait();
        update();
        float t = now_s(), g = 0;
        bool hg = accel_g(g);
        if (L.n_att != att_seen) { att_seen = L.n_att; t_att = t; }
        float h = alt() - h0;
        log_row(n, phase, t - t_ready, h0, spot, hg, g);
        float rate = rate_norm();
        if (has_rel) {
            if (rate > peak_rate) peak_rate = rate;
            if (h > apex) apex = h;
            if (!has_low || h < low) { low = h; has_low = true; }
            tilted = tilted || tilt_deg() > 30;
        }
        /* ---- READY: wait for push, then free fall */
        if (phase == PH_READY) {
            if (!L.armed) {
                beep(TUNE_REFUSED);
                P("Disarmed while READY (CH9 or CH7) - back to WAIT");
                return false;
            }
            if (strcmp(L.mode, "GUIDED_NOGPS")) {
                beep(TUNE_REFUSED);
                P("Mode changed to %s while READY - back to WAIT", L.mode);
                return false;
            }
            if (rc(THROW_CH) < CH_LOW) {
                disarm();
                beep(TUNE_REFUSED);
                P("CH%d OFF - cancelled, disarmed. Back to WAIT", THROW_CH);
                return false;
            }
            if (hg && g > PUSH_G) { t_push = t; has_push = true; }
            if (has_push && t - t_push < PUSH_TO_FREE_S && hg) {
                n_ff = g < FREEFALL_G ? n_ff + 1 : 0;
                if (n_ff >= FREEFALL_N) {
                    t_rel = t - (FREEFALL_N - 1) / HZ;
                    has_rel = true;
                    alt_rel = alt();
                    phase = PH_SPOOL;
                    spool_t0 = t;
                    P("Throw detected - spooling motors");
                }
            }
            continue;
        }
        /* ---- in the air: aborts first */
        if (!L.armed) { why = "disarmed (CH7)"; break; }
        if (strcmp(L.mode, "GUIDED_NOGPS")) { snprintf(why_buf, sizeof(why_buf), "you changed the mode to %s", L.mode); why = why_buf; break; }
        if (rc(THROW_CH) < CH_LOW) { why = "CH5 OFF"; break; }
        if (h > MAX_UP_M) { why = "more than 5 m above the hand"; break; }
        if (t - t_att > STALE_S) { why = "no attitude data from the FC"; break; }
        /* ---- spool -> recover -> caught -> calm (same rules as throw_check) */
        if (phase == PH_SPOOL && (landed_state() != LANDED_ON_GROUND || t - spool_t0 > SPOOL_MAX_S)) {
            t_motors = t;
            has_motors = true;
            if (HAS_LIMITS) { send_limits(true); limits_off = true; }          /* rotation limits off */
            send_params1("GUID_OPTIONS", (float)(guid | GUID_THRUST_BIT));     /* thrust = real thrust */
            phase = PH_RECOVER;
        }
        if (!has_upright && tilted && has_motors && tilt_deg() < TILT_OK) { t_upright = t; has_upright = true; }
        if (phase == PH_RECOVER && rate < RATE_OK && tilt_deg() < TILT_OK && L.vel[2] < 1.0f) {
            if (limits_off) { send_limits(false); limits_off = false; }
            phase = PH_CAUGHT;
            t_caught = t;
            has_caught = true;
        }
        if (phase == PH_RECOVER && has_motors && t - t_motors > RECOVER_MAX_S) { why = "not caught in 3 s"; break; }
        if (has_motors) {
            bool ok = rate < RATE_OK && tilt_deg() < TILT_OK && hspeed() < HSPEED_OK && fabsf(L.vel[2]) < VZ_OK;
            if (ok) { if (!has_held) { held = t; has_held = true; } }
            else has_held = false;
            if (has_held && t - held >= HELD_S) { t_rec = held; has_rec = true; why = "recovered"; break; }
        }
    }
    /* ---- the script's control ends: parameters back FIRST, then GUIDED hold 10 s -> LAND */
    if (limits_off) send_limits(false);
    After after;
    if (L.armed && !strcmp(L.mode, "GUIDED_NOGPS")) after = hold_land(n, t_ready, h0, spot, why);
    send_params1("GUID_OPTIONS", (float)guid);
    if (strcmp(why, "recovered")) P("[stop] %s", why);
    info.n = n;
    info.detected = has_motors;
    info.has_motors = has_motors; info.motors_on_s = t_motors - t_rel;
    info.upright_zero = has_motors && !tilted;
    info.has_upright = has_motors && tilted && has_upright; info.upright_s = t_upright - t_rel;
    info.has_caught = has_caught; info.caught_s = t_caught - t_rel;
    info.recovered = has_rec; info.has_rec = has_rec; info.recovered_s = t_rec - t_rel;
    strncpy(info.ended, why, sizeof(info.ended) - 1);
    info.ended[sizeof(info.ended) - 1] = 0;
    info.apex = apex; info.has_low = has_low; info.low = low; info.peak_rate = peak_rate;
    float dx = L.pos[0] - spot[0], dy = L.pos[1] - spot[1];
    info.drift = sqrtf(dx * dx + dy * dy);
    info.volt = volts();
    info.hover = hover;
    date_str(info.date, sizeof(info.date));
    info.after = after;
    return true;
}

static void save_result(const Info &e, int n)
{
    char name[64], s[700], ck[300] = "";
    snprintf(name, sizeof(name), "%s.csv", BASE);
    int fh = H->file_open(name, true);
    if (fh < 0) { P("[save] cannot open %s", name); return; }
    if (n == 1)
        H->file_write(fh, "n,mode,date,detected,motors_on_s,upright_s,caught_s,recovered,recovered_s,ended,apex_m,low_m,"
                          "peak_rate_rads,drift_m,volt,hover_used,hold_mode,brake_s,brake_m,wander_m,landed_disarmed,"
                          "note,steps_dropped,checks\n");
    for (int k = 0; k < 9; k++) {
        size_t l = strlen(ck);
        snprintf(ck + l, sizeof(ck) - l, "%s%s=%s", k ? "; " : "", CHECKS[k].name, CHECKS[k].txt);
    }
    char mo[12] = "", up[12] = "", ca[12] = "", re[12] = "", lo[12] = "", bs[12] = "", bm[12] = "", wm[12] = "";
    if (e.has_motors) snprintf(mo, 12, "%.2f", e.motors_on_s);
    if (e.has_upright) snprintf(up, 12, "%.2f", e.upright_s); else if (e.upright_zero) snprintf(up, 12, "0.0");
    if (e.has_caught) snprintf(ca, 12, "%.2f", e.caught_s);
    if (e.has_rec) snprintf(re, 12, "%.2f", e.recovered_s);
    if (e.has_low) snprintf(lo, 12, "%.2f", e.low);
    const After &a = e.after;
    if (a.has && a.has_stop) { snprintf(bs, 12, "%.2f", a.brake_s); snprintf(bm, 12, "%.2f", a.brake_m); snprintf(wm, 12, "%.2f", a.wander_m); }
    snprintf(s, sizeof(s), "%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.2f,%s,%.2f,%.2f,%.2f,%.3f,%s,%s,%s,%s,%s,\"%s\",%lu,\"%s\"\n",
             e.n, mode_str(), e.date, e.detected ? "True" : "False", mo, up, ca, e.recovered ? "True" : "False", re,
             e.ended, e.apex, lo, e.peak_rate, e.drift, e.volt, e.hover, a.has ? a.hold_mode : "", bs, bm, wm,
             a.has ? (a.landed_disarmed ? "True" : "False") : "", a.note, (unsigned long)rows_dropped, ck);
    H->file_write(fh, s);
    H->file_close(fh);
}

/* ---------------------------------------------------------------- check mode (ladder step 0) */
static void run_check(float hover)
{
    P("CHECK - never arms, never changes a mode or parameter. Type stop to end.");
    P("Flip CH%d OFF and ON -> 'switch OK' + the check list. With CH%d ON, toss it ~20 cm onto a pillow -> "
      "'throw detected (check only)'. Walking / shaking -> nothing.", THROW_CH, THROW_CH);
    uint32_t imu_n0 = L.n_imu, att_n0 = L.n_att;
    float t_rate = now_s(), imu_hz = 0, att_hz = 0, t_print = 0, t_push = 0;
    bool has_push = false, saw_low = false, link_ok = false;
    int n_ff = 0, ch_was = rc(THROW_CH);
    float g_min = 9, g_max = 0;
    float last[3] = {0, 0, 1};
    char cmd[64];
    rk_start();
    for (;;) {
        rk_wait();
        update();
        if (H->command(cmd, sizeof(cmd)) && !strcmp(cmd, "stop")) { P("check stopped"); return; }
        float g = 0, t = now_s();
        bool hg = accel_g(g);
        int ch = rc(THROW_CH);
        if (hg) { if (g < g_min) g_min = g; if (g > g_max) g_max = g; }
        if (t - t_rate >= 1.0f) {
            imu_hz = (L.n_imu - imu_n0) / (t - t_rate);
            att_hz = (L.n_att - att_n0) / (t - t_rate);
            imu_n0 = L.n_imu; att_n0 = L.n_att; t_rate = t;
        }
        if (!link_ok && imu_hz >= 45 && att_hz >= 45) { link_ok = true; P("link OK  (IMU %.0f Hz, ATT %.0f Hz)", imu_hz, att_hz); }
        if (ch < CH_LOW) saw_low = true;
        if (ch > CH_HIGH && ch_was <= CH_HIGH) {
            if (saw_low) P("switch OK  (CH%d OFF %d- / ON %d+, now %d)", THROW_CH, CH_LOW, CH_HIGH, ch);
            static const char *PN[3] = {"MOT_SPIN_ARM", "MOT_SPOOL_TIME", "RC5_OPTION"};
            get_params(PN, 3);
            bool ok = checks();
            P("buzzer test: 3 SHORT beeps = READY ...");
            beep(TUNE_READY);
            wait_ms(2000);
            P("            ... 2 LONG beeps = REFUSED");
            beep(TUNE_REFUSED);
            P("checks: %s  (arming is your CH9 in a real run)", ok ? "ALL PASS" : "would REFUSE");
            for (int k = 0; k < 9; k++) P("   %s %-30s %s", CHECKS[k].ok ? "ok " : "NO ", CHECKS[k].name, CHECKS[k].txt);
        }
        ch_was = ch;
        if (ch > CH_HIGH && hg) {
            if (g > PUSH_G) { t_push = t; has_push = true; }
            n_ff = (g < FREEFALL_G && has_push && t - t_push < PUSH_TO_FREE_S) ? n_ff + 1 : 0;
            if (n_ff == FREEFALL_N)
                P("throw detected (check only - motors NOT started)  push %.2f g, free fall %.2f g", g_max, g_min);
        }
        if (t - t_print >= 1.0f) {
            int fix, sats;
            gps(fix, sats);
            char ppo[48] = "";
            if (PPO_MODE && L.have_pos) {
                float q[4], thr, l2[3] = {last[0], last[1], last[2]};
                uint32_t t0 = H->us();
                ppo_step(hover, alt(), l2, q, thr);
                snprintf(ppo, sizeof(ppo), " | PPO %.1f ms", (H->us() - t0) / 1000.0f);
            } else if (PPO_MODE) {
                snprintf(ppo, sizeof(ppo), " | PPO - (no EKF position yet)");
            }
            P("CH%d %d CH6 %d CH3 %d | GPS fix %d %d sats | %.2f V | IMU %.0f Hz ATT %.0f Hz | g %.2f-%.2f | tilt %.0f%s",
              THROW_CH, ch, rc(6), rc(3), fix, sats, volts(), imu_hz, att_hz, g_min, g_max, tilt_deg(), ppo);
            g_min = 9; g_max = 0; t_print = t;
        }
    }
}

/* while waiting for the FC / GPS (on the ground, nothing flying): the file commands work too,
   so the results can be copied at home without GPS (and without the FC switched on) */
static bool startup_wait(uint32_t ms)
{
    char cmd[64];
    wait_ms(ms);
    if (!H->command(cmd, sizeof(cmd))) return false;
    if (!strcmp(cmd, "mode ppo") || !strcmp(cmd, "mode rule")) {
        H->save_mode(!strcmp(cmd, "mode ppo"));
        P("mode saved: %s - restarting", !strcmp(cmd, "mode ppo") ? "ppo-policy" : "attitude-rule");
        return true;
    }
    if (!strcmp(cmd, "help")) P("commands now: ls | get <file> | getall | rmall | info | mode ppo | mode rule (check: after READY)");
    else H->file_command(cmd);
    return false;
}

/* ---------------------------------------------------------------- main (section 1) */
void core_run(Hal &h, const CoreConfig &cfg)
{
    H = &h;
    PPO_MODE = cfg.ppo;
    L = Link();
    N_PARAMS = 0;
    P("PPO (C, int16 weights) loaded: %s", PPO_SOURCE);
    P("[link] connecting to the FC (UART)");
    uint32_t t_end = h.ms() + 30000;
    while (!L.have_hb && (int32_t)(h.ms() - t_end) < 0)
        if (startup_wait(20)) return;
    if (!L.have_hb) { P("STOP - no heartbeat in 30 s: FC on? SERIAL2 = MAVLink2 921600? TX-RX crossed?"); return; }
    P("[link] heartbeat from system %d", L.tsys);
    request_rate(MAVLINK_MSG_ID_LOCAL_POSITION_NED, HZ);
    request_rate(MAVLINK_MSG_ID_ATTITUDE_QUATERNION, HZ);
    bool got = false;
    for (int k = 0; k < 8 && !got; k++) {                        /* EKF gives a position only after GPS */
        t_end = h.ms() + 15000;
        while ((int32_t)(h.ms() - t_end) < 0 && !(L.have_q && L.have_pos))
            if (startup_wait(20)) return;
        got = L.have_q && L.have_pos;
        if (!got) P("waiting for the EKF position (GPS fix) ...");
    }
    if (!got) { P("STOP - No position from the EKF after 2 min - check GPS (CH6 middle at power-up)"); return; }
    request_rate(MAVLINK_MSG_ID_RAW_IMU, 50);
    request_rate(MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 10);
    request_rate(MAVLINK_MSG_ID_RC_CHANNELS, 20);
    for (int i = 0; i < 3; i++) request_rate(DATA_ID[i], 2);
    request_rate(MAVLINK_MSG_ID_SYSTEM_TIME, 1);                  /* the date for the result files */
    t_end = h.ms() + 20000;
    uint32_t t_ask = h.ms();
    auto have_all = []() { return L.has[M_IMU] && L.has[M_ESS] && L.has[M_RC] && L.has[M_GPS] && L.has[M_SYS] && L.has[M_EKF]; };
    while ((int32_t)(h.ms() - t_end) < 0 && !have_all()) {
        update();
        if (h.ms() - t_ask > 2000) { ask_data_again(); t_ask = h.ms(); }
        h.sleep_ms(5);
    }
    if (!have_all()) {
        P("STOP - The FC does not send %s%s%s%s%s%s- check SERIAL2 (2 / 921)", L.has[M_EKF] ? "" : "EKF_STATUS_REPORT ",
          L.has[M_ESS] ? "" : "EXTENDED_SYS_STATE ", L.has[M_GPS] ? "" : "GPS_RAW_INT ", L.has[M_IMU] ? "" : "RAW_IMU ",
          L.has[M_RC] ? "" : "RC_CHANNELS ", L.has[M_SYS] ? "" : "SYS_STATUS ");
        return;
    }
    static const char *PN[5] = {"MOT_SPIN_ARM", "MOT_SPOOL_TIME", "RC5_OPTION", "GUID_OPTIONS", "MOT_THST_HOVER"};
    get_params(PN, 5);
    float hover = 0.149f, spin = 1, spool = 0.5, rc5 = -1, guid_f = 0;
    param_get("MOT_THST_HOVER", hover);
    bool hs = param_get("MOT_SPIN_ARM", spin), hp = param_get("MOT_SPOOL_TIME", spool), hr = param_get("RC5_OPTION", rc5);
    param_get("GUID_OPTIONS", guid_f);
    char a1[12], a2[12], a3[12];
    snprintf(a1, 12, hs ? "%.1f" : "None", spin); snprintf(a2, 12, hp ? "%.2f" : "None", spool); snprintf(a3, 12, hr ? "%.1f" : "None", rc5);
    P("Connected | %s | hover thrust %.3f | MOT_SPIN_ARM %s | MOT_SPOOL_TIME %s | RC5_OPTION %s", mode_str(), hover, a1, a2, a3);
    if (hs && fabsf(spin) < 1e-6f && hp && spool > 0.3f)
        P("WARNING: MOT_SPIN_ARM 0 with MOT_SPOOL_TIME > 0.3 - motors start ~0.6 s after release (SITL 8 Nov: "
          "roll throws drifted 23 m). See guide C3: MOT_SPOOL_TIME 0.25");
    /* values left from a stopped run (power pulled mid-throw) -> put back */
    int guid = (int)guid_f;
    for (int s = 0; s < 2 && !HAS_LIMITS; s++) {
        get_params(LIMIT_NAMES[s], 3);
        float v[3];
        if (param_get(LIMIT_NAMES[s][0], v[0]) && param_get(LIMIT_NAMES[s][1], v[1]) && param_get(LIMIT_NAMES[s][2], v[2])) {
            HAS_LIMITS = true;
            LIMITS_N = LIMIT_NAMES[s];
            memcpy(LIMITS_V, v, sizeof(v));
        }
    }
    if (guid & GUID_THRUST_BIT) {
        guid &= ~GUID_THRUST_BIT;
        send_params1("GUID_OPTIONS", (float)guid);
        P("[fix] values left from a stopped run - restored: GUID_OPTIONS %d", guid);
    }
    for (int i = 0; HAS_LIMITS && i < 3; i++) {
        if (fabsf(LIMITS_V[i]) < 1e-6f) {
            int k = -1;
            for (int j = 0; j < 3; j++) if (!strcmp(LIMITS_N[i], REAL_LIMIT_NAMES[j])) k = j;
            if (k < 0) { P("STOP - %s = 0 left from a stopped run: set it back in Mission Planner", LIMITS_N[i]); return; }
            LIMITS_V[i] = REAL_LIMIT_VALUES[k];
            send_params1(LIMITS_N[i], LIMITS_V[i]);
            P("[fix] values left from a stopped run - restored: %s %.0f", LIMITS_N[i], LIMITS_V[i]);
        }
    }
    /* outputs: run<boot>.csv / run<boot>_steps.csv / run<boot>.json */
    snprintf(BASE, sizeof(BASE), "run%04lu", (unsigned long)cfg.boot_no);
    char name[64], s[600];
    snprintf(name, sizeof(name), "%s.json", BASE);
    int fh = h.file_open(name, false);
    if (fh >= 0) {
        snprintf(s, sizeof(s), "{\"boot\": %lu, \"mode\": \"%s\", \"hover\": %.3f, \"model\": \"%s\", \"board\": \"ESP32-S3\", "
                 "\"throw_ch\": %d, \"hold_s\": %.1f, \"max_up_m\": %.1f, \"freefall_g\": %.2f, \"freefall_n\": %d, "
                 "\"push_g\": %.2f, \"min_volt\": %.1f, \"min_sats\": %d, \"limits\": {\"%s\": %.0f, \"%s\": %.0f, \"%s\": %.0f}, "
                 "\"guid_options\": %d, \"mot_spool_time\": %.2f}\n", (unsigned long)cfg.boot_no, mode_str(), hover,
                 PPO_MODE ? PPO_SOURCE : "", THROW_CH, HOLD_S, MAX_UP_M, FREEFALL_G, FREEFALL_N, PUSH_G, MIN_VOLT, MIN_SATS,
                 HAS_LIMITS ? LIMITS_N[0] : "none", LIMITS_V[0], HAS_LIMITS ? LIMITS_N[1] : "none", LIMITS_V[1],
                 HAS_LIMITS ? LIMITS_N[2] : "none", LIMITS_V[2], guid, spool);
        h.file_write(fh, s);
        h.file_close(fh);
    }
    int n = 0, ch_was = rc(THROW_CH);
    bool steps_header = false;
    float t_print = 0;
    beep(PPO_MODE ? TUNE_START_PPO : TUNE_START_RULE);
    P("ESP32 READY TO USE (%s: %s beep)", mode_str(), PPO_MODE ? "1 long + 2 short" : "1 long + 1 short");
    P("WAIT - CH8 ALT_HOLD, stick LOW (keep it low), arm with CH9 (props still), then CH%d ON. Type help for commands.", THROW_CH);
    char cmd[64];
    for (;;) {
        update();
        int ch = rc(THROW_CH);
        if (ch > CH_HIGH && ch_was <= CH_HIGH) {                  /* throw switch just went ON */
            static const char *PN3[3] = {"MOT_SPIN_ARM", "MOT_SPOOL_TIME", "RC5_OPTION"};
            get_params(PN3, 3);
            if (!checks()) {
                beep(TUNE_REFUSED);
                char r[300] = "REFUSED: ";
                for (int k = 0; k < 9; k++)
                    if (!CHECKS[k].ok) {
                        size_t l = strlen(r);
                        snprintf(r + l, sizeof(r) - l, "%s%s (%s)", l > 9 ? "; " : "", CHECKS[k].name, CHECKS[k].txt);
                    }
                h.out(r);
            } else {
                Info e;
                snprintf(name, sizeof(name), "%s_steps.csv", BASE);
                steps_fh = h.file_open(name, true);
                if (steps_fh >= 0 && !steps_header)
                    h.file_write(steps_fh, "n,mode,phase,t_s,tilt_deg,rate_rads,height_above_hand_m,hspeed_ms,dist_from_hand_m,"
                                           "accel_g,landed_state,ch5,p_rads,q_rads,r_rads\n");
                steps_header = true;
                if (one_throw(n + 1, hover, guid, e)) {
                    n++;
                    flush_rows(ROWS_MAX);
                    if (steps_fh >= 0) h.file_close(steps_fh);
                    steps_fh = -1;
                    save_result(e, n);                            /* after EVERY throw */
                    char mo[10] = "-", ca[10] = "-", re[10] = "-", lo[10] = "-", bm[10] = "None", wm[10] = "None";
                    if (e.has_motors) snprintf(mo, 10, "%.2f", e.motors_on_s);
                    if (e.has_caught) snprintf(ca, 10, "%.2f", e.caught_s);
                    if (e.has_rec) snprintf(re, 10, "%.2f", e.recovered_s);
                    if (e.has_low) snprintf(lo, 10, "%.2f", e.low);
                    if (e.after.has && e.after.has_stop) { snprintf(bm, 10, "%.2f", e.after.brake_m); snprintf(wm, 10, "%.2f", e.after.wander_m); }
                    P("Throw %d | %s | motors on %s s | caught %s s | recovered %s s | lowest %s m (hand = 0) | peak rate %.1f | "
                      "drift %.2f m | %s -> %s | braked %s m, held within %s m", n, mode_str(), mo, ca, re, lo, e.peak_rate,
                      e.drift, e.ended, e.after.has ? e.after.hold_mode : "None", bm, wm);
                    P("Saved: %s.csv / _steps.csv / .json", BASE);
                    P("Next throw: CH%d OFF, arm again (CH9), CH%d ON.", THROW_CH, THROW_CH);
                } else {
                    rows_head = 0; rows_count = 0;                /* nothing thrown: no steps saved */
                    if (steps_fh >= 0) h.file_close(steps_fh);
                    steps_fh = -1;
                }
            }
            back_to_alt_hold();                                   /* version d, change 1 */
            ch = rc(THROW_CH);
        }
        ch_was = ch;
        if (now_s() - t_print > 2.0f) {
            ask_data_again();                                     /* version d, change 2 */
            int fix, sats;
            gps(fix, sats);
            P("WAIT | CH%d %d CH6 %d CH3 %d | GPS fix %d %d sats | %.2f V | %s %s%s", THROW_CH, ch, rc(6), rc(3), fix, sats,
              volts(), L.armed ? "ARMED" : "disarmed", L.mode, ch > CH_HIGH ? "  <- CH5 is ON: flip it OFF, then ON to start" : "");
            t_print = now_s();
        }
        if (h.command(cmd, sizeof(cmd))) {                        /* typed commands: only when disarmed */
            if (L.armed) P("commands only when disarmed");
            else if (!strcmp(cmd, "check")) run_check(hover);
            else if (!strcmp(cmd, "mode ppo") || !strcmp(cmd, "mode rule")) {
                h.save_mode(!strcmp(cmd, "mode ppo"));
                P("mode saved: %s - restarting", !strcmp(cmd, "mode ppo") ? "ppo-policy" : "attitude-rule");
                return;
            } else if (!strcmp(cmd, "help")) {
                P("commands: check | stop (ends check) | mode ppo | mode rule | ls | get <file> | getall | rmall | info");
            } else h.file_command(cmd);
        }
        wait_ms(20);
    }
}
