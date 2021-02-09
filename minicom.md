## Microcontroller Terminal

**Project description:** This project interfaces with an emulated (due to distance learning) AT91SAM9263 microcontroller. Using the minicom terminal, it allows users to input commands based on a simple hierarchical menu system. Example commands include turning on or off LEDs, blinking LEDs for 5 seconds, getting DBGU status, or setting the buttons.
 
### Implementation

```c
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "AT91SAM9263.h"

#define PIO_PORT_B AT91C_BASE_PIOB
#define PIO_PORT_C AT91C_BASE_PIOC

#define DBGU_RECEIVER 1<<30
#define DBGU_TRANSMITTER 1<<31
#define DBGU AT91C_BASE_DBGU
#define RESET_RECEIVER AT91C_US_RSTRX
#define RESET_TRANSMITTER AT91C_US_RSTTX
#define ENABLE_RECEIVER AT91C_US_RXEN 
#define ENABLE_TRANSMITTER AT91C_US_TXEN
#define RXRDY AT91C_US_RXRDY
#define TXRDY AT91C_US_TXRDY 
#define ENDRX AT91C_US_ENDRX
#define ENDTX AT91C_US_ENDTX 
#define OVRE AT91C_US_OVRE
#define FRAME AT91C_US_FRAME
#define PARE AT91C_US_PARE
#define TXEMPTY AT91C_US_TXEMPTY
#define TXBUFE AT91C_US_TXBUFE
#define RXBUFF AT91C_US_RXBUFF
#define COMMTX AT91C_US_COMM_TX
#define COMMRX AT91C_US_COMM_RX
#define MCK 100000000
#define BAUD_RATE 19200
#define BAUD_GENERATOR (unsigned int) MCK/(16*BAUD_RATE)
#define CHANNEL_MODE AT91C_US_CHMODE_AUTO
#define PARITY_TYPE AT91C_US_PAR_NONE
#define CLEAR_MINICOM_STR "\x1b[2J"
#define BUFFER_SIZE 50

#define LEDA AT91C_PIO_PB8
#define LEDB AT91C_PIO_PC29
#define BLINK_DELAY 1000

#define RIGHT_BUTTON AT91C_PIO_PC4
#define LEFT_BUTTON AT91C_PIO_PC5

#define CLOCK AT91C_BASE_PMC
#define PORTS_C_TO_E 1<<4

#define PIT AT91C_BASE_PITC
#define PIT_ENABLE AT91C_PITC_PITEN
#define PIT_INTERRUPT_ENABLE AT91C_PITC_PITIEN
#define PIT_CYCLE_MS 10
#define CLOCK_RATE 100000
#define COUNTER_RATE 16
#define PIT_PIV_OVERFLOW (CLOCK_RATE*PIT_CYCLE_MS)/COUNTER_RATE - 1

#define AIC AT91C_BASE_AIC 
#define SYS_CONTROLLER_ID AT91C_ID_SYS

enum menu {start, led, button, dbgu} currentMenu;
int pitCounter;
int otherCounter;

int initDbgu(void) {
	//disable interrupts
	DBGU->DBGU_IDR = TXRDY | RXRDY | ENDRX | ENDTX | OVRE | FRAME | PARE | TXEMPTY | TXBUFE | RXBUFF | COMMTX | COMMRX;
	//reset transceiver
	DBGU->DBGU_CR = RESET_RECEIVER | RESET_TRANSMITTER;
	//configure as peripheral ports
	PIO_PORT_C->PIO_ASR = DBGU_RECEIVER | DBGU_TRANSMITTER;
	PIO_PORT_C->PIO_PDR = DBGU_RECEIVER | DBGU_TRANSMITTER;
	//configure throughput
	DBGU->DBGU_BRGR = BAUD_GENERATOR;
	//setup communications type
	DBGU->DBGU_MR = CHANNEL_MODE | PARITY_TYPE;
	//enable transceiver
	DBGU->DBGU_CR = ENABLE_RECEIVER | ENABLE_TRANSMITTER;
	return 0;
}

void initLeds() {
	PIO_PORT_B->PIO_PER = LEDA;
	PIO_PORT_B->PIO_OER = LEDA;
	PIO_PORT_B->PIO_SODR = LEDA;
	PIO_PORT_C->PIO_PER = LEDB;
	PIO_PORT_C->PIO_OER = LEDB;
	PIO_PORT_C->PIO_SODR = LEDB;
}

void initButtons(void) {
	CLOCK->PMC_PCER = PORTS_C_TO_E;
	PIO_PORT_C->PIO_PER = RIGHT_BUTTON | LEFT_BUTTON;
	PIO_PORT_C->PIO_ODR = RIGHT_BUTTON | LEFT_BUTTON;
	PIO_PORT_C->PIO_PPUER = RIGHT_BUTTON | LEFT_BUTTON;
}

void IrqHandler(void) {
	if (PIT->PITC_PIMR & PIT_INTERRUPT_ENABLE) { //check if interrupt is set
		if (PIT->PITC_PISR) { //check if PIV has been reached
			PIT->PITC_PIVR; //clear PIV
			pitCounter++;
		}
		else {
			otherCounter++; //it was another device's interrupt
		}
	}
}

void initPIT(void) {
	PIT->PITC_PIMR = PIT_INTERRUPT_ENABLE | PIT_PIV_OVERFLOW;
}

void initAIC(void) {
	AIC->AIC_IDCR = 1<<SYS_CONTROLLER_ID;
	AIC->AIC_SVR[SYS_CONTROLLER_ID] = (int) IrqHandler;
	AIC->AIC_SMR[SYS_CONTROLLER_ID] = AT91C_AIC_SRCTYPE_INT_LEVEL_SENSITIVE | AT91C_AIC_PRIOR_LOWEST;
	AIC->AIC_ICCR = 1<<SYS_CONTROLLER_ID;
	AIC->AIC_IECR = 1<<SYS_CONTROLLER_ID;
}

void printCharDbgu(const unsigned char c) {
	while((DBGU->DBGU_CSR & TXRDY) == 0) {};
	DBGU->DBGU_THR = c;
}

void readDataDbgu(unsigned char *data) {
	while ((DBGU->DBGU_CSR & RXRDY) == 0) {};
	*data = (unsigned char) DBGU->DBGU_RHR;
}

void readStringDbgu(unsigned char *data) {
	unsigned char c;
	unsigned int i = 0;
	memset(data, 0, BUFFER_SIZE);
	do {
		while ((DBGU->DBGU_CSR & RXRDY) == 0) {};
		c = (unsigned char) DBGU->DBGU_RHR;
		if (c >= 'A' && c <= 'Z') {
			data[i++] = c + 'a'-'A';
			printCharDbgu(c);
		}
		else if (c >= 'a' && c <= 'z') {
			data[i++] = c;
			printCharDbgu(c);
		}
		else if (c == ' ') {
			data[i++] = c;
			printCharDbgu(c);
		}
		if (i >= BUFFER_SIZE) {
			return;
		}
	} while (c != '\r' && c != '\n' && c != 0);
}

void printStringDbgu(const unsigned char *string) {
	int i;
	for (i=0;i<strlen(string);i++) {
		printCharDbgu(string[i]);
	}
}

void resetMinicom() {
	printStringDbgu(CLEAR_MINICOM_STR);
	printStringDbgu("Simple menu by Maya Gillilan\n\n\tWrite 'help' for more information\n\n\tLED>\n\n\tBUTTON>\n\n\tDBGU>\n\n\t>");
	currentMenu = start;
}

void goToLedMenu() {
	printStringDbgu(CLEAR_MINICOM_STR);
	printStringDbgu("LED\n\n\tSetLED\n\n\tClearLED\n\n\tBlinkLED\n\n\tLEDStatus\n\n\tChangeLED\n\n\t>");
	currentMenu = led;
}

void goToButtonMenu() {
	printStringDbgu(CLEAR_MINICOM_STR);
	printStringDbgu("BUTTON\n\n\tReadButton\n\n\tPullupEn\n\n\tPullupDis\n\n\t>");
	currentMenu = button;
}

void goToDbguMenu() {
	printStringDbgu(CLEAR_MINICOM_STR);
	printStringDbgu("DBGU\n\n\tDeviceStatus\n\n\t>");
	currentMenu = dbgu;
}

int PITDelay(int delayMs) {
	PIT->PITC_PIMR |= PIT_ENABLE;
	while ((pitCounter*PIT_CYCLE_MS) < delayMs) {
	}
	pitCounter = 0;
	PIT->PITC_PIMR ^= PIT_ENABLE;
	return 0;
}

void setLed(char label) {
	if (label == 'a') {
		PIO_PORT_B->PIO_CODR = LEDA;
	}
	else if (label == 'b') {
		PIO_PORT_C->PIO_CODR = LEDB;
	}
}

void clearLed(char label) {
	if (label == 'a') {
		PIO_PORT_B->PIO_SODR = LEDA;
	}
	else if (label == 'b') {
		PIO_PORT_C->PIO_SODR = LEDB;
	}
}

void blinkLed(char label) {
	int i;
	if (label == 'a') {
		for (i=0;i<5;i++) {
			PIO_PORT_B->PIO_CODR = LEDA;
			PITDelay(BLINK_DELAY);
			PIO_PORT_B->PIO_SODR = LEDA;
			PITDelay(BLINK_DELAY);
		}
	}
	else if (label == 'b') {
		for (i=0;i<5;i++) {
			PIO_PORT_C->PIO_CODR = LEDB;
			PITDelay(BLINK_DELAY);
			PIO_PORT_C->PIO_SODR = LEDB;
			PITDelay(BLINK_DELAY);
		}
	}
}

void LedStatus(char label) {
	if (label == 'a') {
		if ((PIO_PORT_B->PIO_ODSR & LEDA) == 0) {
			printStringDbgu("\nLED A is ON");
		}
		else {
			printStringDbgu("\nLED A is OFF");
		}
	}
	else if (label == 'b') {
		if ((PIO_PORT_C->PIO_ODSR & LEDB) == 0) {
			printStringDbgu("\nLED B is ON");
		}
		else {
			printStringDbgu("\nLED B is OFF");
		}
	}
}

void changeLed(char label) {
	if (label == 'a') {
		if ((PIO_PORT_B->PIO_ODSR & LEDA) == 0) {
			PIO_PORT_B->PIO_SODR = LEDA;
		}
		else {
			PIO_PORT_B->PIO_CODR = LEDA;
		}
	}
	else if (label == 'b') {
		if ((PIO_PORT_C->PIO_ODSR & LEDB) == 0) {
			PIO_PORT_C->PIO_SODR = LEDB;
		}
		else {
			PIO_PORT_C->PIO_CODR = LEDB;
		}
	}
}

void readButton(char label) {
	if (label == 'a') {
		if ((PIO_PORT_C->PIO_PDSR & LEFT_BUTTON) == 0) {
			printStringDbgu("\n Left button is PRESSED");
		}
		else {
			printStringDbgu("\n Left button is NOT PRESSED");
		}
	}
	if (label == 'b') {
		if ((PIO_PORT_C->PIO_PDSR & RIGHT_BUTTON) == 0) {
			printStringDbgu("\n Right button is PRESSED");
		}
		else {
			printStringDbgu("\n Right button is NOT PRESSED");
		}
	}
}

void pullupEn(char label) {
	if (label == 'a') {
		PIO_PORT_C->PIO_PPUER = LEFT_BUTTON;
	}
	if (label == 'b') {
		PIO_PORT_C->PIO_PPUER = RIGHT_BUTTON;
	}
}

void pullupDis(char label) {
	if (label == 'a') {
		PIO_PORT_C->PIO_PPUDR = LEFT_BUTTON;
	}
	if (label == 'b') {
		PIO_PORT_C->PIO_PPUDR = RIGHT_BUTTON;
	}
}

void printDbguStatus() {
	printStringDbgu("\nBaudrate: ");
	int dbguBufferSize = 20;
	char tmp[dbguBufferSize];
	sprintf(tmp, "%d", BAUD_RATE);
	printStringDbgu(tmp);
	memset(tmp, 0, dbguBufferSize);
	sprintf(tmp,"%d",(MCK/(DBGU->DBGU_BRGR*16)));
	printStringDbgu("\nCalculated baudrate: ");
	printStringDbgu(tmp);
	printStringDbgu("\nData bits: 8\nParity bits: ");
	int tmpi = DBGU->DBGU_MR;
	if ((tmpi & 1<<11) == 1<<11) {
		printStringDbgu("0");
	}
	if (((tmpi & (1<<15 | 1<<14)) == 1<<14)) {
		printStringDbgu("\nChannel Mode: Auto");
	}
}

void checkCommand(unsigned char *data) {
	if (strcmp(data,"led") == 0) {
		goToLedMenu();
		return;
	}
	else if (strcmp(data,"button") == 0) {
		goToButtonMenu();
		return;
	}
	else if (strcmp(data,"dbgu") == 0) {
		goToDbguMenu();
		return;
	}
	else if (strcmp(data,"up") == 0) {
		resetMinicom();
		return;
	}
	else if (strcmp(data,"help") == 0) {
		if (currentMenu == start) {
			printStringDbgu("\nType the desired submenu name and then press enter to navigate. \n\t>");
		}
		else if (currentMenu == led) {
			printStringDbgu("\nType a command followed by A or B for the desired LED, then press enter.\n");
			printStringDbgu("SetLED: Turn on an LED.\nClearLED: Turn off an LED.\nBlinkLED: Blink an LED 5 times.\n");
			printStringDbgu("LEDStatus: Check if an LED is on or off.\nChangeLED: Invert an LED.\n");
			printStringDbgu("To return, type \"up\" and press enter.\n\t>");
		}
		else if (currentMenu == button) {
			printStringDbgu("\nType a command followed by A or B for the desired button, then press enter.\n");
			printStringDbgu("ReadButton: Get a button's status\nPullupEn: Enable a pull-up register\nPullupDis: Disable a pull-up register\n");
			printStringDbgu("To return, type \"up\" and press enter.\n\t>");
		}
		else if (currentMenu == dbgu) {
			printStringDbgu("\nType a command followed by enter.");
			printStringDbgu("DeviceStatus: Get information about the DBGU device.\n");
			printStringDbgu("To return, type \"up\" and press enter.\n\t>");
		}
		return;
	}
	if (currentMenu == led) {
		if (strcmp(data,"setled a") == 0 || strcmp(data, "setled b") == 0) {
			setLed(data[strlen(data)-1]);
		}
		else if (strcmp(data, "clearled a") == 0 || strcmp(data, "clearled b") == 0) {
			clearLed(data[strlen(data)-1]);
		}
		else if (strcmp(data, "blinkled a") == 0 || strcmp(data, "blinkled b") == 0) {
			blinkLed(data[strlen(data)-1]);
		}
		else if (strcmp(data, "ledstatus a") == 0 || strcmp(data, "ledstatus b") == 0) {
			LedStatus(data[strlen(data)-1]);
		}
		else if (strcmp(data, "changeled a") == 0 || strcmp(data, "changeled b") == 0) {
			changeLed(data[strlen(data)-1]);
		}
	}
	else if (currentMenu == button) {
		if (strcmp(data, "readbutton a") == 0 || strcmp(data, "readbutton b") == 0) {
			readButton(data[strlen(data)-1]);
		}
		else if (strcmp(data, "pullupen a") == 0 || strcmp(data, "pullupen b") == 0) {
			pullupEn(data[strlen(data)-1]);
		}
		else if (strcmp(data, "pullupdis a") == 0 || strcmp(data, "pullupdis b") == 0) {
			pullupDis(data[strlen(data)-1]);
		}
	}
	else if (currentMenu == dbgu) {
		if (strcmp(data, "devicestatus") == 0) {
			printDbguStatus();
		}
	}
	printStringDbgu("\n\t>");
}


int main(void) {
	initDbgu();
	initLeds();
	initButtons();
	initPIT();
	initAIC();
	unsigned char data[BUFFER_SIZE];
	resetMinicom();
	while(1) {
		readStringDbgu(data);
		checkCommand(data);
	}
}

```
