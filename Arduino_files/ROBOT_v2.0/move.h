

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
  
int F_R_RPWM = 10;
int F_R_LPWM = 0;
  
int B_L_RPWM = 0;
int B_L_LPWM = 0;
  
int B_R_RPWM = 0;
int B_R_LPWM = 0;


void move_pin_init(){
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


int Wheel_move(int move_state, int linear_x, int angular_z){
  int state = 0;
  if (move_state == 1) {
    state = 1;
    
    F_L_RPWM = 0;
    F_L_LPWM = 0;
    F_R_RPWM = 0;
    F_R_LPWM = 0;
    B_L_RPWM = 0;
    B_L_LPWM = 0;
    B_R_RPWM = 0;
    B_R_LPWM = 0;
   
  }
  else {
    if (linear_x > 0){      
      F_L_RPWM = 0;
      F_L_LPWM = linear_x;
      
      F_R_RPWM = linear_x;
      F_R_LPWM = 0;
      
      B_L_RPWM = linear_x;
      B_L_LPWM = 0;
      
      B_R_RPWM = linear_x;
      B_R_LPWM = 0;

      if ( angular_z > 0) {
        F_R_RPWM = F_R_RPWM + angular_z;
        if ( F_L_LPWM - angular_z > 0){
          F_L_LPWM = F_L_LPWM - angular_z;
        }
        else {
          F_L_LPWM = 0;
        }
        B_R_RPWM = B_R_RPWM + angular_z;
        if ( B_L_RPWM - angular_z > 0) {
          B_L_RPWM = B_L_RPWM - angular_z;
        }
        else {
          B_L_RPWM = 0;
        }
      }
      else if ( angular_z < 0){
        F_L_LPWM = F_L_LPWM - angular_z;
        if ( F_R_RPWM + angular_z > 0) {
          F_R_RPWM = F_R_RPWM + angular_z;
        }
        else {
          F_R_RPWM = 0;
        }        
        B_L_RPWM = B_L_RPWM - angular_z;
        if ( B_R_RPWM + angular_z > 0) {
          B_R_RPWM = B_R_RPWM + angular_z;
        }
        else {
          B_R_RPWM = 0;
        }
        
      }
      
    }
    else if ( linear_x < 0) {      
      F_L_RPWM = -linear_x;
      F_L_LPWM = 0;
      
      F_R_RPWM = 0;
      F_R_LPWM = -linear_x;
      
      B_L_RPWM = 0;
      B_L_LPWM = -linear_x;
      
      B_R_RPWM = 0;
      B_R_LPWM = -linear_x;

      if ( angular_z > 0) {
        F_R_LPWM = F_R_LPWM + angular_z;
        B_R_LPWM = B_R_LPWM + angular_z;
      }
      else if ( angular_z < 0){
        F_L_RPWM = F_L_RPWM - angular_z;
        B_L_LPWM = B_L_LPWM - angular_z;
      }
      
    }
    else {
      F_L_RPWM = 0;
      F_L_LPWM = 0; 
      F_R_RPWM = 0;
      F_R_LPWM = 0;
      B_L_RPWM = 0;
      B_L_LPWM = 0;
      B_R_RPWM = 0;
      B_R_LPWM = 0;

      if ( angular_z > 0) {
        F_L_RPWM = angular_z * 255/100;
        F_L_LPWM = 0;
        
        F_R_RPWM = angular_z * 255/100;
        F_R_LPWM = 0;
        
        B_L_RPWM = 0;
        B_L_LPWM = angular_z * 255/100;
        
        B_R_RPWM = angular_z * 255/100;
        B_R_LPWM = 0;
      }
      else if ( angular_z < 0) {
        F_L_RPWM = 0;
        F_L_LPWM = -angular_z * 255/100;
        
        F_R_RPWM = 0;
        F_R_LPWM = -angular_z * 255/100;
        
        B_L_RPWM = -angular_z * 255/100;
        B_L_LPWM = 0;
        
        B_R_RPWM = 0;
        B_R_LPWM = -angular_z * 255/100;
      }
    }

    
  }

  digitalWrite( F_L_DRIVER_EN, HIGH);
  digitalWrite( F_R_DRIVER_EN, HIGH);
  digitalWrite( B_L_DRIVER_EN, HIGH);
  digitalWrite( B_R_DRIVER_EN, HIGH);

  analogWrite( F_L_DRIVER_RPWM , F_L_RPWM);
  analogWrite( F_L_DRIVER_LPWM , F_L_LPWM);
    
  analogWrite( F_R_DRIVER_LPWM , F_R_LPWM);
  analogWrite( F_R_DRIVER_RPWM , F_R_RPWM );
      
  analogWrite( B_L_DRIVER_RPWM , B_L_RPWM);
  analogWrite( B_L_DRIVER_LPWM , B_L_LPWM);
    
  analogWrite( B_R_DRIVER_LPWM , B_R_LPWM);
  analogWrite( B_R_DRIVER_RPWM , B_R_RPWM);
    
}
