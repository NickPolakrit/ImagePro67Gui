/*
 * File:   assignmentImage.c
 * Author: G_Peav
 *
 * Created on October 16, 2019, 4:31 PM
 */


#include "xc.h"
#include <stdio.h>
#include "configuration.h"
#include "UART_rev2.h"
#define FCY 40008571        // 40008571.428571.... Hz
#define cal 123


void initPLL() {
    PLLFBD = 150; // M  = 152
    CLKDIVbits.PLLPRE = 5; // N1 = 7
    CLKDIVbits.PLLPOST = 0; // N2 = 2
    OSCTUN = 0; // Tune FRC oscillator, if FRC is used

    // Clock switching to incorporate PLL
    __builtin_write_OSCCONH(0x01); // Initiate Clock Switch to FRCPLL
    // Oscillator with PLL (NOSC=0b011)
    __builtin_write_OSCCONL(0x01); // Start clock switching

    while (OSCCONbits.COSC != 0b001); // Wait for Clock switch to occur
    while (OSCCONbits.LOCK != 1) {
    }; // Wait for PLL to lock
}

int get_new_value() {
    printf("%c", 0xFF);
    printf("%c", 0xFF);
    printf("%c", 0x01);
    printf("%c", 0x01);
    printf("%c", 0xFD);
    int t, c;
    while (UART1_ReceiveBufferSizeGet() == 0) {
    }
    t = UART1_Read();
    while (UART1_ReceiveBufferSizeGet() == 0) {
    }
    c = UART1_Read();
    UART1_Write(c);
    if (t == 1) {
        return c;
    } else {
        return 0 - c;
    }
}

int send_value(int value) {
    printf("%c", 0xFF);
    printf("%c", 0xFF);
    printf("%c", 0x02);
    printf("%c", value);
    printf("%c", 0xFF - (0x02 + value));
    int t, c;
    while (UART1_ReceiveBufferSizeGet() == 0) {
    }
    t = UART1_Read();
    while (UART1_ReceiveBufferSizeGet() == 0) {
    }
    c = UART1_Read();
    UART1_Write(c);
    if (t == 1) {
        return c;
    } else {

        return 0 - c;
    }
}

int main(void) {
    /*disable global interrupt*/
    __builtin_disable_interrupts();

    initPLL();
    __builtin_write_OSCCONL(OSCCON & 0xBF); //PPS RECONFIG UNLOCK 
    RPINR18bits.U1RXR = 6;
    RPOR2bits.RP5R = 0b00011;
    __builtin_write_OSCCONL(OSCCON | 0x40); //PPS RECONFIG LOCK 

    /* UARTx is disabled, Flow Control Mode, UxRX idle state is '1', 
     * High-speed mode, 8-bit data, no parity, 1 stop bit
     * wait at least 1 usec (1/1000000) before sending first char
     * First Argument is U1MODE register
     * Second Argument is U1STA register
     * Third Argument is U1BRG register
     * Fourth Argument is waiting time before sending first char
     */
    UART1_Initialize(0x0008, 0x0000, ((FCY / 1000000) / 4) - 1, 40);

    /*enable global interrupt*/
    __builtin_enable_interrupts();

    unsigned char input[cal][cal];
    int i = 0;
    int j = 0;
    int k = 0;
    int h = 0;
    int row;
    int col;
    int result = 0;
    int x = 0;
    int y = 0;  // 9 variable
    row = cal ;
    col = cal ;

    for (i = 0; i < cal; i++) {
        for (j = 0; j < cal; j++) {
            input[i][j] = get_new_value();
        }
    }
    
    int c = 0; 
    c = cal / 3;

    for (i = 0; i < c; i++) {
        for (j = 0; j < c; j++) {
            for (k = 0; k < 3; k++) {
                for (h = 0; h < 3; h++) {
                    x = row + k;
                    y = col + h;
                    result += (int) input[x][y];
                }
            }
            row -= 3;
            send_value(result / 9);
            result = 0;
        }
        col = col - 3;
        row = 120;
    }

    return 0;
}