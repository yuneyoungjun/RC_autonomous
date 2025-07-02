# utils/pid_controller.py

class PID:
    def __init__(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.last_error = 0
        self.integral = 0
        self.setpoint = 0 # 목표값을 설정할 수도 있지만, 여기서는 오차를 직접 받으므로 생략 가능
        self.count=0
        self.anti_windup=ki*0.1
        self.line=5
        self.alpha=0.8

    # ? PID 게인 값을 동적으로 설정하는 메서드 추가
    def set_gains(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd

    def pid_control(self, error, dt):
        # P 항
        self.count+=1
        proportional = self.Kp * error

        # I 항 (적분 누적)
        # if self.count>40:
        #     self.integral=0
        if error>self.line:
            self.integral-=self.anti_windup*(error-self.line)*dt
        self.integral += error * dt
        self.integral*=self.alpha
        # 적분 항의 오버슈트 방지를 위한 제한 (예시)
        # self.integral = max(min(self.integral, 100), -100) 

        # D 항
        derivative = (error - self.last_error) / dt
        self.last_error = error

        # 최종 제어 출력
        output = proportional + (self.Ki * self.integral) + (self.Kd * derivative)
        return output