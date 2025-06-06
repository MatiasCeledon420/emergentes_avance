using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

[System.Serializable]
public class Sign {
    public string text;
    public Sprite sprite;
    // public GameObject hand;

    public Sign(Sign sign) {
        text   = sign.text;
        sprite = sign.sprite;
    }

    public Sign Clone() {
        return new Sign(this);
    }
}

public class State : MonoBehaviour {
    [SerializeField] public Sign[] signs; 

    public static State data;

    void Awake() {
        if ( data == null ) {
            data = this;
            DontDestroyOnLoad(gameObject);
        } else if ( data != this ) {
            Destroy(gameObject);
        }
    }
}