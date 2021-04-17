
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;
using VRC.Udon.Common.Interfaces;

public class SendEvent : UdonSharpBehaviour
{
    public UdonSharpBehaviour Behaviour;
    public string EventName;

    public int _currentSky =31;

    void Start()
    {
        if (_currentSky == null) _currentSky = (int)Behaviour.GetProgramVariable("_syncedSky");
        
    }

     void Interact()
    {
        Debug.Log("Sending Event: " + EventName);
        // handler.SetProgramVariable("playerID", playerID);
        // handler.SetProgramVariable("newEvent", eventString);
        // handler.SendCustomEvent("Handle");

        
        _currentSky++;
        
//        Debug.Log("Setting ProgramVariable: _syncedSky to " + _currentSky);
//        Behaviour.SetProgramVariable("_newSky",_currentSky);

        Debug.Log("Sending Custom Event: " + EventName);
        Behaviour.SendCustomEvent(EventName);
        
        Debug.Log("Event: " + EventName +" has been sent");

//GetComponent (UdonBehaviour) -> SendCustomEvent

//Then the set value one has to have the gameobject and not the behaviour

//        Target.SendEvent(EventName); //.SetActive(!Target.activeInHierarchy);
    }

}
