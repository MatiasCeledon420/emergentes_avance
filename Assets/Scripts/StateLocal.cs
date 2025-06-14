using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class StateLocal : MonoBehaviour {
    public GameObject rsButton;
    public Sprite rsRed;
    public Sprite rsGreen;

    public TMP_InputField goalInput;

    void Start() {
        if ( State.data ) {
            Change_Random_Mode();
        }
    }

    public void Change_Random_Mode() {
        State.data.randomSelection = !State.data.randomSelection;

        if ( !rsButton ) { return; }

        rsButton.GetComponent<Image>().sprite = ( State.data.randomSelection )
            ? rsGreen
            : rsRed;
    }

    public void Change_Goal(string value) {
        State.data.goal = ( !string.IsNullOrEmpty(value) ) ? int.Parse(value) : int.MaxValue;
    }

    public void Change_Time(string value) {
        State.data.time = ( !string.IsNullOrEmpty(value) ) ? int.Parse(value) : int.MaxValue;
    }

    public void Set_Total_Goal() {
        State.data.goal = State.data.signs.Where(iter => iter.enable).ToList().Count;
        goalInput.text  = State.data.goal.ToString();
    }
}