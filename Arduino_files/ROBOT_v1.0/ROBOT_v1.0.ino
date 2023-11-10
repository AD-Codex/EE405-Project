
#define F_L_DRIVER_EN 28
#define F_L_DRIVER_RPWM 8
#define F_L_DRIVER_LPWM 9

#define F_R_DRIVER_EN 22
#define F_R_DRIVER_RPWM 2
#define F_R_DRIVER_LPWM 3

#define B_L_DRIVER_EN 26
#define B_L_DRIVER_RPWM 6
#define B_L_DRIVER_LPWM 7

#define B_R_DRIVER_EN 24
#define B_R_DRIVER_RPWM 4
#define B_R_DRIVER_LPWM 5

// L high at left move forward

int F_L_RPWM = 0;
int F_L_LPWM = 0;

int F_R_RPWM = 0;
int F_R_LPWM = 0;

int B_L_RPWM = 0;
int B_L_LPWM = 0;

int B_R_RPWM = 0;
int B_R_LPWM = 0;


void setup() {
  // put your setup code here, to run once:

  Serial.begin(115200);
  
  pinMode( F_L_DRIVER_EN , OUTPUT);
  pinMode( F_L_DRIVER_RPWM , OUTPUT);
  pinMode( F_L_DRIVER_LPWM , OUTPUT);

  pinMode( F_R_DRIVER_EN , OUTPUT);
  pinMode( F_R_DRIVER_RPWM , OUTPUT);
  pinMode( F_R_DRIVER_LPWM , OUTPUT);

  pinMode( B_L_DRIVER_EN , OUTPUT);
  pinMode( B_L_DRIVER_RPWM , OUTPUT);
  pinMode( B_L_DRIVER_LPWM , OUTPUT);

  pinMode( B_R_DRIVER_EN , OUTPUT);
  pinMode( B_R_DRIVER_RPWM , OUTPUT);
  pinMode( B_R_DRIVER_LPWM , OUTPUT);

  digitalWrite( F_L_DRIVER_EN, LOW);
  digitalWrite( F_R_DRIVER_EN, LOW);
  digitalWrite( B_L_DRIVER_EN, LOW);
  digitalWrite( B_R_DRIVER_EN, LOW);

}

void loop() {
  // put your main code here, to run repeatedly:

//  digitalWrite( F_L_DRIVER_EN, HIGH);
//  analogWrite( F_L_DRIVER_RPWM , 0);
//  analogWrite( F_L_DRIVER_LPWM , 200);
//
//  digitalWrite( B_L_DRIVER_EN, HIGH);
//  analogWrite( B_L_DRIVER_RPWM , 0);
//  analogWrite( B_L_DRIVER_LPWM , 200);

  if ( Serial.available() > 0){
    int readByte = Serial.read();
    Serial.println(readByte);

    if ( readByte == 87) {
      Serial.println("move_forward");
      move_forward(50);
    }
    else if ( readByte == 83 ) {
      Serial.println("move_stop");
      move_stop();
    }
    else if ( readByte == 88 ) {
      Serial.println("move_back");
      move_back(50);
    }
  }

  delay(100);

}

void move_forward( int move_speed) {
  
  analogWrite( F_L_DRIVER_LPWM , 0);
  analogWrite( F_L_DRIVER_RPWM , 0);

  analogWrite( F_R_DRIVER_RPWM , 0);
  analogWrite( F_R_DRIVER_LPWM , 0);
  
  analogWrite( B_L_DRIVER_LPWM , 0);
  analogWrite( B_L_DRIVER_RPWM , 0);

  analogWrite( B_R_DRIVER_RPWM , 0);
  analogWrite( B_R_DRIVER_LPWM , 0);
  

  int pwm_speed = 0;

  if ( move_speed >=0 && move_speed <=100) {
    pwm_speed = move_speed*(255/100);
  }

  F_L_RPWM = 0;
  F_L_LPWM = pwm_speed;
  F_R_RPWM = pwm_speed;
  F_R_LPWM = 0;
  B_L_RPWM = 0;
  B_L_LPWM = pwm_speed;
  B_R_RPWM = pwm_speed;
  B_R_LPWM = 0;

  analogWrite( F_L_DRIVER_RPWM , F_L_RPWM);
  analogWrite( F_L_DRIVER_LPWM , F_L_LPWM);

  analogWrite( F_R_DRIVER_LPWM , F_R_LPWM);
  analogWrite( F_R_DRIVER_RPWM , F_R_RPWM );
  
  analogWrite( B_L_DRIVER_RPWM , B_L_RPWM);
  analogWrite( B_L_DRIVER_LPWM , B_L_LPWM);

  analogWrite( B_R_DRIVER_LPWM , B_R_LPWM);
  analogWrite( B_R_DRIVER_RPWM , B_R_RPWM);

  digitalWrite( F_L_DRIVER_EN, HIGH);
  digitalWrite( F_R_DRIVER_EN, HIGH);
  digitalWrite( B_L_DRIVER_EN, HIGH);
  digitalWrite( B_R_DRIVER_EN, HIGH);
  
}


void move_back( int move_speed) {

  analogWrite( F_L_DRIVER_LPWM , 0);
  analogWrite( F_L_DRIVER_RPWM , 0);

  analogWrite( F_R_DRIVER_RPWM , 0);
  analogWrite( F_R_DRIVER_LPWM , 0);
  
  analogWrite( B_L_DRIVER_LPWM , 0);
  analogWrite( B_L_DRIVER_RPWM , 0);

  analogWrite( B_R_DRIVER_RPWM , 0);
  analogWrite( B_R_DRIVER_LPWM , 0);
  

  int pwm_speed = 0;

  if ( move_speed >=0 && move_speed <=100) {
    pwm_speed = move_speed*(255/100);
  }

  F_L_RPWM = pwm_speed;
  F_L_LPWM = 0;
  F_R_RPWM = 0;
  F_R_LPWM = pwm_speed;
  B_L_RPWM = pwm_speed;
  B_L_LPWM = 0;
  B_R_RPWM = 0;
  B_R_LPWM = pwm_speed;

  analogWrite( F_L_DRIVER_LPWM , F_L_LPWM);
  analogWrite( F_L_DRIVER_RPWM , F_L_RPWM);

  analogWrite( F_R_DRIVER_RPWM , F_R_RPWM);
  analogWrite( F_R_DRIVER_LPWM , F_R_LPWM);
  
  analogWrite( B_L_DRIVER_LPWM , B_L_LPWM);
  analogWrite( B_L_DRIVER_RPWM , B_L_RPWM);

  analogWrite( B_R_DRIVER_RPWM , B_R_RPWM);
  analogWrite( B_R_DRIVER_LPWM , B_R_LPWM);

  digitalWrite( F_L_DRIVER_EN, HIGH);
  digitalWrite( F_R_DRIVER_EN, HIGH);
  digitalWrite( B_L_DRIVER_EN, HIGH);
  digitalWrite( B_R_DRIVER_EN, HIGH);
  
}


void move_stop(){

  analogWrite( F_L_DRIVER_LPWM , 0);
  analogWrite( F_L_DRIVER_RPWM , 0);

  analogWrite( F_R_DRIVER_RPWM , 0);
  analogWrite( F_R_DRIVER_LPWM , 0);
  
  analogWrite( B_L_DRIVER_LPWM , 0);
  analogWrite( B_L_DRIVER_RPWM , 0);

  analogWrite( B_R_DRIVER_RPWM , 0);
  analogWrite( B_R_DRIVER_LPWM , 0);

  analogWrite( F_L_DRIVER_LPWM , F_L_RPWM);
  analogWrite( F_L_DRIVER_RPWM , F_L_LPWM);

  analogWrite( F_R_DRIVER_RPWM , F_R_LPWM);
  analogWrite( F_R_DRIVER_LPWM , F_R_RPWM);
  
  analogWrite( B_L_DRIVER_LPWM , B_L_RPWM);
  analogWrite( B_L_DRIVER_RPWM , B_L_LPWM);

  analogWrite( B_R_DRIVER_RPWM , B_R_LPWM);
  analogWrite( B_R_DRIVER_LPWM , B_R_RPWM);

  delay(500);

  analogWrite( F_L_DRIVER_LPWM , 0);
  analogWrite( F_L_DRIVER_RPWM , 0);

  analogWrite( F_R_DRIVER_RPWM , 0);
  analogWrite( F_R_DRIVER_LPWM , 0);
  
  analogWrite( B_L_DRIVER_LPWM , 0);
  analogWrite( B_L_DRIVER_RPWM , 0);

  analogWrite( B_R_DRIVER_RPWM , 0);
  analogWrite( B_R_DRIVER_LPWM , 0);

  digitalWrite( F_L_DRIVER_EN, LOW);
  digitalWrite( F_R_DRIVER_EN, LOW);
  digitalWrite( B_L_DRIVER_EN, LOW);
  digitalWrite( B_R_DRIVER_EN, LOW);

}
