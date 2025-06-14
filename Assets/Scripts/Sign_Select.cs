using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

// using Tinter;

public class Sign_Select : MonoBehaviour {
    public GameObject elem;
    public Transform gridContainer;

    private List<GameObject> instances = new List<GameObject>();

    void Start() {
        DisplayArray();
    }

    void DisplayArray() {
        int index = 0;

        foreach (Sign sign in State.data.signs) {
            GameObject signObj = Instantiate(elem, gridContainer);
            instances.Add(signObj);

            var copy = index;
            signObj.GetComponent<Button>().onClick.AddListener(() => Change_Enable(copy));
            signObj.transform.Find("Image").GetComponent<Image>().sprite = sign.sprite;
            signObj.transform.Find("Text").GetComponent<TextMeshProUGUI>().text = sign.text;

            Tinter.Hierarchy(signObj.transform, ( sign.enable ) 
                ? new Color(1f, 1f, 1f, 1f)
                : new Color(0.5f, 0.5f, 0.5f, 1f)
            );

            index += 1;
        }
    }

    public Color InverseColor(Color c) {
        return new Color(
            c.r != 0 ? 1f / c.r : 0f,
            c.g != 0 ? 1f / c.g : 0f,
            c.b != 0 ? 1f / c.b : 0f,
            c.a != 0 ? 1f / c.a : 0f
        );
    }

    public void Change_Enable(int index) {
        Sign sign = State.data.signs[index];
        sign.enable = !sign.enable;

        // Debug.Log($"{index}: {sign.enable}");

        var color = new Color(0.5f, 0.5f, 0.5f, 1f);
        
        Tinter.Hierarchy(instances[index].transform, ( sign.enable ) 
            ? InverseColor(color)
            : color
        );
    }

    public void Invert_Selection() {
        for (int index = 0; index < State.data.signs.Length; index += 1) {
            Change_Enable(index);
        }
    }
}