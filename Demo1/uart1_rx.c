/*
 * File:   uart1.c
 * Author: G_Peav
 *
 * Created on September 9, 2019, 4:31 PM
 */


#include "xc.h"
#include <stdio.h>
#include "configuration.h"
#include "UART.h"
#define FCY 40008571        // 40008571.428571.... Hz
int numByte;
uint8_t dataArray[10];
void initPLL()
{
    PLLFBD = 150;           // M  = 152
    CLKDIVbits.PLLPRE = 5;  // N1 = 7
    CLKDIVbits.PLLPOST = 0; // N2 = 2
    OSCTUN = 0;             // Tune FRC oscillator, if FRC is used
    
    // Clock switching to incorporate PLL
    __builtin_write_OSCCONH(0x01);    // Initiate Clock Switch to FRCPLL
    // Oscillator with PLL (NOSC=0b011)
    __builtin_write_OSCCONL(0x01);    // Start clock switching

    while (OSCCONbits.COSC != 0b001); // Wait for Clock switch to occur
    while (OSCCONbits.LOCK!=1) {};    // Wait for PLL to lock
}

int main(void) {
    /*disable global interrupt*/
    __builtin_disable_interrupts();
    
    TRISBbits.TRISB10 = 0;
    TRISBbits.TRISB11 = 0;
    TRISBbits.TRISB12 = 0;
    TRISBbits.TRISB13 = 0;
    TRISBbits.TRISB14 = 0;
    
    
    initPLL();
    __builtin_write_OSCCONL(OSCCON & 0xBF); //PPS RECONFIG UNLOCK 
    RPINR18bits.U1RXR = 6;
    RPOR2bits.RP5R = 0b00011;
    __builtin_write_OSCCONL(OSCCON | 0x40); //PPS RECONFIG LOCK 
    
    
    /* UARTx is disabled, Flow Control Mode, UxRX idle state is '1', 
     * High-speed mode, 8-bit data, no parity, 1 stop bit
     * wait at least 9 usec (1/115200) before sending first char
     */
    UART1_Initialize(0x0008, 0x0000, ((FCY/115200)/4)-1, 360);

    /*enable global interrupt*/
    __builtin_enable_interrupts();
    
    //--------------------------------------------------------------------------
    printf("------------");
//    commandS[5] = {"0", "0", "0", "0", "0"};
    while(1)
    {
        numByte = UART1_ReadBuffer(dataArray, 10);
        
        if (numByte != 0){
            // Receive data
            LATBbits.LATB10 = 0;
            LATBbits.LATB11 = 0;
            LATBbits.LATB12 = 0;
            LATBbits.LATB13 = 0;
            LATBbits.LATB14 = 0;
            
            if(dataArray[0] == '1' ){
                printf("%d\n",numByte);
                LATBbits.LATB10 = 1;
            }
            else if(dataArray[0] == '2' ){
                printf("%d\n",numByte);
                LATBbits.LATB11 = 1;
            }
            else if(dataArray[0] == '3' ){
                printf("%d\n",numByte);
                LATBbits.LATB12 = 1;
            }
            else if(dataArray[0] == '4' ){
                printf("%d\n",numByte);
                LATBbits.LATB13 = 1;
            }
            else if(dataArray[0] == '5' ){
                printf("%d\n",numByte);
                LATBbits.LATB14 = 1;
            }
        }
    }
    return 0;
}